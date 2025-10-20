# ollama run llama3:instruct

import os
import json
import glob
import re
from typing import List, Dict

from tqdm import tqdm
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# === Global Configuration ===
OLLAMA_MODEL = "llama3:instruct"
INPUT_DIR = "math_feedback_data"
OUTPUT_DIR = "evaluated_feedback_json"

SYSTEM_MESSAGE = (
    "You are an educational expert evaluating mentor feedback given on student math work. "
    "Each file contains a math problem, a student's submission, and one or more rounds of mentor feedback. "
    "Evaluate ONLY the mentor feedback, using the rubric provided.\n\n"
    "Rubric Categories:\n"
    "1. Clarity – Is the feedback easy to understand?\n"
    "2. Specificity – Does the mentor identify precise strengths and weaknesses?\n"
    "3. Constructiveness – Does the feedback help the student improve?\n"
    "4. Tone – Is the mentor respectful and encouraging?\n"
    "5. Cognitive Demand – Does the feedback prompt reflection or deeper revision?\n\n"
    "Instructions:\n"
    "- Assign one score (1–4) per category.\n"
    "- Provide a single paragraph justification citing examples from the mentor feedback.\n"
    "- Your output must be a JSON object with a 'scores' dictionary and a 'justification' string.\n"
)

HUMAN_TEMPLATE = (
    "Document:\n"
    "{input}\n\n"
    "Please provide a JSON response of the form:\n"
    "{{\n"
    "  \"scores\": {{\n"
    "    \"Clarity\": int,\n"
    "    \"Specificity\": int,\n"
    "    \"Constructiveness\": int,\n"
    "    \"Tone\": int,\n"
    "    \"Cognitive Demand\": int\n"
    "  }},\n"
    "  \"justification\": str\n"
    "}}\n"
)

# === File Discovery ===
def find_txt_files(directory: str) -> List[str]:
    return glob.glob(os.path.join(directory, "**", "*.txt"), recursive=True)

# === Prompt Construction ===
def build_prompt_chain():
    system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE)
    human_prompt = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)
    return ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# === Evaluation Pipeline ===
def evaluate_file(filepath: str, llm, chain) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        response = chain.invoke({"input": content})
        raw_output = response['content'] if isinstance(response, dict) else str(response)

        match = re.search(r"{.*}", raw_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            return {"error": "No JSON found in response", "raw": raw_output}
    except Exception as e:
        return {"error": str(e)}

# === Main Function with tqdm ===
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    llm = OllamaLLM(model=OLLAMA_MODEL)
    chain = build_prompt_chain() | llm

    files = find_txt_files(INPUT_DIR)

    with tqdm(files, desc="Evaluating mentor feedback", unit="file") as pbar:
        for filepath in pbar:
            pbar.set_postfix(file=os.path.basename(filepath)[:20])
            result = evaluate_file(filepath, llm, chain)

            output_filename = os.path.splitext(os.path.basename(filepath))[0] + ".json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            with open(output_path, 'w', encoding='utf-8') as out_file:
                json.dump(result, out_file, indent=2)

    print(f"\nFinished evaluating {len(files)} files. Results saved to '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
