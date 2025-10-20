# Full Pipeline: Fine-Tuning Embedding + LLM and Re-Embedding Chroma Batches

import os
import json
from pathlib import Path
from uuid import uuid4
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from chromadb import PersistentClient
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# === SAMPLE JSON FORMATS ===
#
# embedding_training.json:
# [
#   {"query": "What is RAG?", "positive": "RAG stands for Retrieval-Augmented Generation."},
#   {"query": "Define vector database.", "positive": "A vector database stores embeddings of documents."}
# ]
#
# llm_training.json:
# [
#   {"instruction": "What is Retrieval-Augmented Generation?", "output": "It is a technique that retrieves documents before generating an answer."},
#   {"instruction": "Explain embeddings in simple terms.", "output": "Embeddings are numerical representations of text used to compare meaning."}
# ]

# === GLOBAL CONFIGURATION ===
CHROMA_BATCHES_DIR = "vectorstore"                   # Existing Chroma vectorstore with batch_* subdirs
COLLECTION_PREFIX = "vectordb"                        # Prefix used for collections
EMBEDDING_TUNE_JSON = "embedding_training.json"       # JSON with query-positive pairs
LLM_TUNE_JSON = "llm_training.json"                   # JSON with instruction-output pairs
EMBEDDING_MODEL_BASE = "intfloat/e5-large-v2"         # Pretrained embedding model
LLM_MODEL_BASE = "tiiuae/falcon-7b-instruct"          # Pretrained LLM for fine-tuning
EMBEDDING_MODEL_TUNED_DIR = "./fine_tuned_embeddings"# Output for fine-tuned embedding model
LLM_MODEL_TUNED_DIR = "./fine_tuned_llm"              # Output for fine-tuned LLM
CHROMA_REWRITE_DIR = "vectorstore_reembedded"         # Output for re-embedded vectorstore

# === Step 1: Fine-Tune Embedding Model ===
def fine_tune_embedding_model():
    print("\n[Step 1] Fine-tuning embedding model...")
    with open(EMBEDDING_TUNE_JSON, 'r') as f:
        data = json.load(f)
    model = SentenceTransformer(EMBEDDING_MODEL_BASE)
    examples = [InputExample(texts=[item['query'], item['positive']], label=1.0) for item in data]
    dataloader = DataLoader(examples, shuffle=True, batch_size=16)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(dataloader, loss)], epochs=1, show_progress_bar=True)
    model.save(EMBEDDING_MODEL_TUNED_DIR)
    print(f"✓ Saved fine-tuned embedding model to: {EMBEDDING_MODEL_TUNED_DIR}")

# === Step 2: Fine-Tune LLM ===
def fine_tune_llm_model():
    print("\n[Step 2] Fine-tuning LLM model (LoRA)...")
    with open(LLM_TUNE_JSON, 'r') as f:
        data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_BASE)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_BASE).to("cuda" if torch.cuda.is_available() else "cpu")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    def format_pair(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return {"input_ids": tokenizer(prompt, truncation=True, max_length=512, padding="max_length")['input_ids']}
    dataset = Dataset.from_list(data).map(format_pair)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        output_dir=LLM_MODEL_TUNED_DIR,
        num_train_epochs=1,
        save_steps=10,
        logging_steps=5,
        save_total_limit=1,
        fp16=torch.cuda.is_available()
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(LLM_MODEL_TUNED_DIR)
    tokenizer.save_pretrained(LLM_MODEL_TUNED_DIR)
    print(f"✓ Saved fine-tuned LLM to: {LLM_MODEL_TUNED_DIR}")

# === Step 3: Re-Embed Documents into ChromaDB ===
def reembed_chroma_batches():
    print("\n[Step 3] Re-embedding Chroma batches with fine-tuned embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_TUNED_DIR)
    batch_dirs = sorted(Path(CHROMA_BATCHES_DIR).glob("batch_*"))
    if not batch_dirs:
        raise FileNotFoundError(f"No Chroma batches found in {CHROMA_BATCHES_DIR}")
    for batch_dir in tqdm(batch_dirs, desc="Re-embedding Batches"):
        old_dir = Path(batch_dir)
        new_dir = Path(CHROMA_REWRITE_DIR) / old_dir.name if CHROMA_REWRITE_DIR else old_dir
        new_dir.mkdir(parents=True, exist_ok=True)
        old_client = PersistentClient(path=str(old_dir))
        old_collection_name = f"{COLLECTION_PREFIX}_{old_dir.name}"
        old_collection = old_client.get_collection(old_collection_name)
        offset = 0
        all_docs, all_metadatas, all_ids = [], [], []
        while True:
            results = old_collection.get(include=["documents", "metadatas"], limit=500, offset=offset)
            if not results["documents"]:
                break
            all_docs.extend(results["documents"])
            all_metadatas.extend(results["metadatas"])
            all_ids.extend(results["ids"])
            offset += 500
        if not all_docs:
            print(f"Skipping {old_dir.name}: no documents found.")
            continue
        new_embeddings = embedding_model.encode(all_docs, convert_to_numpy=True, show_progress_bar=True)
        new_client = PersistentClient(path=str(new_dir))
        new_collection_name = f"{COLLECTION_PREFIX}_{old_dir.name}"
        new_collection = new_client.get_or_create_collection(new_collection_name)
        try:
            new_collection.delete()
            new_collection = new_client.get_or_create_collection(new_collection_name)
        except Exception as e:
            print(f"Warning: Could not delete existing collection in {new_dir}: {e}")
        new_collection.add(
            documents=all_docs,
            metadatas=all_metadatas,
            ids=all_ids,
            embeddings=new_embeddings
        )
        print(f"Re-embedded batch {old_dir.name} → {new_dir}")
    print("\nAll batches re-embedded.")
    if CHROMA_REWRITE_DIR:
        print(f"New Chroma vectorstore written to: {CHROMA_REWRITE_DIR}")
    else:
        print("Existing Chroma vectorstore overwritten with re-embedded documents.")

# === MAIN PIPELINE ===
if __name__ == "__main__":
    fine_tune_embedding_model()
    fine_tune_llm_model()
    reembed_chroma_batches()
    print("\nFull RAG fine-tuning pipeline completed.")
