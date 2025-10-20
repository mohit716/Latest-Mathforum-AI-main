# pip install gradio transformers peft bitsandbytes accelerate torch
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

import os
import sys
import traceback
from pathlib import Path
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel

# === Configuration ===
LLM_MODEL_PATH = "./llama3-finetuned"  # adapter dir OR merged full-model dir OR HF repo id
BASE_MODEL = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"  # required if LLM_MODEL_PATH is an adapter dir
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 512
DO_SAMPLE = True
REPETITION_PENALTY = 1.1
BASE_PROMPT = "Generate appropriate math feedback for the student."

# === Utilities ===
def eprint(prefix: str, e: Exception):
    print(f"[{prefix}] {e}", file=sys.stderr)
    traceback.print_exc()

def _is_local_dir(p: str | Path) -> bool:
    return Path(p).exists() and Path(p).is_dir()

def _is_adapter_dir(p: str | Path) -> bool:
    """Heuristic: PEFT/LoRA adapter directories usually contain adapter_config.json and/or adapter_model.safetensors."""
    d = Path(p)
    if not _is_local_dir(d):
        return False
    names = {x.name for x in d.iterdir()}
    return ("adapter_config.json" in names) or any(n.endswith(".safetensors") and "adapter" in n for n in names)

def _has_tokenizer_assets(p: str | Path) -> bool:
    d = Path(p)
    if not _is_local_dir(d):
        return False
    names = {x.name for x in d.iterdir()}
    return (
        "tokenizer.json" in names or
        "tokenizer_config.json" in names or
        "vocab.json" in names or
        "merges.txt" in names or
        any(n.endswith(".model") for n in names)  # sentencepiece
    )

# === Model/Tokenizer Loading (adapter-aware) ===
def load_tokenizer_and_model(model_ref: str, base_model_ref: str | None = None):
    """
    If model_ref is a local adapter directory, load BASE model and apply adapter.
    Else, treat model_ref as a full model (local or HF repo id).
    Returns (tokenizer, model).
    """
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        if _is_adapter_dir(model_ref):
            if not base_model_ref:
                raise RuntimeError(
                    "Detected a PEFT/LoRA adapter directory but BASE_MODEL is not set."
                )

            # Tokenizer: prefer adapter dir if it has tokenizer assets; otherwise use base
            if _has_tokenizer_assets(model_ref):
                tokenizer = AutoTokenizer.from_pretrained(model_ref, legacy=True, use_fast=False)
            else:
                tokenizer = AutoTokenizer.from_pretrained(base_model_ref, legacy=True, use_fast=False)

            # Base model (quantized) + apply adapter
            base = AutoModelForCausalLM.from_pretrained(
                base_model_ref,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, model_ref)
            print("[load] Loaded base model and applied PEFT adapter.")

        else:
            # Full model or HF repo id
            tokenizer = AutoTokenizer.from_pretrained(model_ref, legacy=True, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            print("[load] Loaded full model/tokenizer from:", model_ref)

        return tokenizer, model

    except Exception as e:
        eprint("load_tokenizer_and_model", e)
        raise

# === Load Fine-Tuned LLM (adapter-aware) ===
print(f"Loading model from {LLM_MODEL_PATH} ...")
tokenizer, model = load_tokenizer_and_model(LLM_MODEL_PATH, base_model_ref=BASE_MODEL)

gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    do_sample=DO_SAMPLE,
    repetition_penalty=REPETITION_PENALTY,
)

# === Gradio Query Function ===
def generate_feedback(user_input: str) -> str:
    prompt = f"{BASE_PROMPT}\n\n{user_input}"
    print(f"=== Prompt ===\n{prompt}\n")
    try:
        result = gen_pipeline(prompt)[0]["generated_text"]
        print(f"=== Output ===\n{result}\n")
        # Remove the prompt prefix if it's echoed in the output
        return result.replace(prompt, "").strip()
    except Exception as e:
        eprint("generate_feedback", e)
        return f"Error: {e}"

# === Gradio UI ===
interface = gr.Interface(
    fn=generate_feedback,
    inputs=gr.Textbox(lines=6, placeholder="Enter a math problem and student response..."),
    outputs="text",
    title="MathForum LLM Feedback Assistant",
    description="Interact with a fine-tuned LLaMA/Mistral model to generate mentor feedback based on student responses.",
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
