# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 
# pip install transformers datasets peft trl bitsandbytes accelerate tqdm blobfile sentencepiece

import os
import json
import zipfile
from pathlib import Path
from typing import List, Dict
import multiprocessing
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel

# === CONFIGURATION ===
ZIP_PATH = "input.zip"
OUTPUT_JSONL = "fine_tuning_data.jsonl"
MODEL_DIR = "llm-finetuned"
BASE_MODEL = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
USE_FAST_TOKENIZER = False # set to true if embedding / fast tokenizer is available
DO_SAMPLE = True
TEMPERATURE = 0.1 # ignored if DO_SAMPLE is false

# === STEP 1: Extract and Convert ===
def extract_json_examples(zip_path: str, output_jsonl: str) -> None:
    examples = []

    def extract_examples_from_thread(thread: Dict) -> List[Dict]:
        thread_examples = []
        problem = thread.get("puzzle_text", "").strip()
        conversation = thread.get("conversation", [])
        for entry in conversation:
            student = entry.get("long_answer", "").strip() or entry.get("short_answer", "").strip()
            mentor = entry.get("response", "").strip()
            if student and mentor:
                input_text = f"Problem: {problem}\nStudent: {student}"
                output_text = mentor
                thread_examples.append({
                    "instruction": "Generate appropriate math feedback for the student.",
                    "input": input_text,
                    "output": output_text
                })
        return thread_examples

    print(f"Reading from ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as archive:
        json_files = [f for f in archive.infolist() if f.filename.endswith(".json") and not f.is_dir()]
        print(f"Found {len(json_files)} JSON files")

        for file_info in tqdm(json_files, desc="Processing JSON files"):
            try:
                with archive.open(file_info) as f:
                    data = json.load(f)
                    examples.extend(extract_examples_from_thread(data))
            except Exception as e:
                print(f"Error processing {file_info.filename}: {e}")

    print(f"Writing {len(examples)} examples to {output_jsonl}")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for ex in tqdm(examples, desc="Writing examples"):
            json.dump(ex, f)
            f.write("\n")

# === STEP 2: Fine-Tune Model ===
def make_tokenize_function(tokenizer, chunk_stride=256):
    max_len = tokenizer.model_max_length

    def tokenize_function(examples):
        new_examples = {"input_ids": [], "attention_mask": []}

        for i, (instruction, input_, output) in enumerate(
            zip(examples["instruction"], examples["input"], examples["output"])
        ):
            full_text = f"{instruction}\n{input_}\n\n{output}"
            tokenized = tokenizer(full_text, truncation=False, add_special_tokens=False)
            seq_len = len(tokenized["input_ids"])

            if seq_len <= max_len:
                encoded = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=max_len,
                    return_tensors=None,
                )
                new_examples["input_ids"].append(encoded["input_ids"])
                new_examples["attention_mask"].append(encoded["attention_mask"])
            else:
                tqdm.write(f"[Chunking] Example {i} has {seq_len} tokens; chunking with stride {chunk_stride}.")

                tokenized_chunks = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=max_len,
                    stride=chunk_stride,
                    return_overflowing_tokens=True,
                    return_tensors=None,
                )

                for j in range(len(tokenized_chunks["input_ids"])):
                    chunk_ids = tokenized_chunks["input_ids"][j]
                    chunk_mask = tokenized_chunks["attention_mask"][j]

                    if isinstance(chunk_ids, torch.Tensor):
                        chunk_ids = chunk_ids.tolist()
                    if isinstance(chunk_mask, torch.Tensor):
                        chunk_mask = chunk_mask.tolist()

                    new_examples["input_ids"].append(chunk_ids)
                    new_examples["attention_mask"].append(chunk_mask)

        return new_examples

    return tokenize_function
    
def fine_tune_model(jsonl_path: str, output_dir: str):
    print("Starting fine-tuning...")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    def format_example(example):
        return {
            "instruction": example["instruction"],
            "input": example["input"],
            "output": example["output"]
        }

    formatted_data = [format_example(ex) for ex in tqdm(data, desc="Formatting examples")]
    dataset = Dataset.from_list(formatted_data)

    # Quantization config (replaces deprecated load_in_4bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=1000,
        save_total_limit=1,
        bf16=True,
        report_to="none"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=USE_FAST_TOKENIZER, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token       

    tokenize_function = make_tokenize_function(tokenizer)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=max(1, multiprocessing.cpu_count())
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

# === STEP 3: Interactive Query ===
def chat_loop(model_dir: str):
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, legacy=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",  # or "fp4"
        bnb_4bit_compute_dtype=torch.float16
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base, model_dir)

    while True:
        user_input = input("\nEnter a math problem and student response:\n")
        if user_input.lower() in {"exit", "quit"}:
            break

        prompt = f"Generate appropriate math feedback for the student.\n{user_input}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Now that 'inputs' exists, you can safely access 'input_ids'
        input_length = inputs['input_ids'].shape[1]
        max_length = model.config.max_position_embeddings
        max_new_tokens = max_length - input_length

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE
        )
        print("\nMentor Feedback:\n" + tokenizer.decode(output[0], skip_special_tokens=True))

# === MAIN FUNCTION ===
def main():
    if os.path.exists(MODEL_DIR):
        print(f"{MODEL_DIR} already exists. Skipping extraction and fine-tuning.")
        chat_loop(MODEL_DIR)
        return

    if not os.path.exists(OUTPUT_JSONL):
        print(f"{OUTPUT_JSONL} not found. Extracting from ZIP...")
        extract_json_examples(ZIP_PATH, OUTPUT_JSONL)
    else:
        print(f"{OUTPUT_JSONL} already exists. Skipping extraction.")

    fine_tune_model(OUTPUT_JSONL, MODEL_DIR)
    chat_loop(MODEL_DIR)

if __name__ == "__main__":
    main()
