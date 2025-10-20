# Running the MathForum AI Toolkit

This document provides a concise walkthrough for setting up a Python environment and running the most commonly used scripts in this repository. It assumes you have access to the MathForum datasets (SQLite/PostgreSQL) and any LLM checkpoints referenced in the scripts.

## 1. Prerequisites

- **Operating system**: Ubuntu 22.04 LTS (or compatible Linux distribution)
- **Hardware**: NVIDIA GPU with at least 16 GB VRAM recommended for LLM fine-tuning and RAG workloads
- **Python**: 3.10 or newer
- **System packages**: `python3-venv`, `build-essential`, `python3-tk`
- **Optional services**:
  - [Ollama](https://ollama.ai) for running local LLMs
  - PostgreSQL 14+ if you plan to use the async data exporters

Install prerequisites on Ubuntu:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential python3-tk
```

## 2. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** If a `requirements.txt` file is not present, inspect the scripts you plan to run and install dependencies individually (e.g., `pip install langchain chromadb fastapi uvicorn textual rich`).

## 3. Prepare Data Sources

1. Place the MathForum SQLite database (e.g., `mathforum.db`) in a known directory, or configure PostgreSQL connection details in your environment variables.
2. Gather any zipped JSON/TXT corpora required by the processing scripts (see the `textualize_*.py` and `filter_text_*.py` modules for expected inputs).
3. Download or build the desired LLM models in Ollama (for example `mistral:7b-instruct-q4_K_M` or `llama3:8b-instruct-q4_K_M`).

## 4. Run Common Workflows

### 4.1 Export Conversations from SQLite

```bash
python textualize_conversation.py \
  --sqlite-path /path/to/mathforum.db \
  --output-dir ./exports/sqlite
```

Key flags:
- `--batch-size`: control chunk size when reading submissions.
- `--include-images`: include decoded image placeholders in the output.

### 4.2 Export Conversations from PostgreSQL

```bash
python textualize_conversation_postgres.py \
  --dsn postgresql://user:pass@host:5432/dbname \
  --output-dir ./exports/postgres
```

This async exporter streams results in parallel; ensure `asyncpg` is installed.

### 4.3 Filter and Score Conversations

```bash
python filter_text_problem_outputs_conversation.py \
  --input-zip ./exports/sqlite/conversations.zip \
  --config ./filter_text_problem_outputs_conversation.json \
  --output-dir ./filtered
```

Outputs include:
- Filtered ZIP archive of JSON/TXT threads
- Wide-format CSVs with rubric deltas and diffs
- Per-rubric CSV/ZIP artifacts

### 4.4 Launch the RAG Web Chat

Start the backend and serve the web UI:

```bash
export OLLAMA_HOST=localhost:11434  # adjust if running multiple Ollama instances
ollama serve &                      # ensure the model is running
ollama run llama3:8b-instruct-q4_K_M &
python mathforum_rag_webchat.py --host 0.0.0.0 --port 7860
```

Visit `http://localhost:7860` in a browser to test the chat interface.

### 4.5 Launch the Fine-Tuned RAG Web Chat

If you have generated a fine-tuned Chroma store and LLM weights, run:

```bash
python mathforum_rag_webchat_finetuned.py \
  --vector-store ./vectordb \
  --ollama-model llama3:8b-instruct-q4_K_M \
  --host 0.0.0.0 --port 7861
```

### 4.6 Fine-Tune the Retrieval-Augmented Pipeline

```bash
python finetune_rag.py \
  --input-json ./filtered/threads.json \
  --vector-store ./vectordb \
  --ollama-model llama3:8b-instruct-q4_K_M
```

Ensure the target Ollama model is already pulled or available locally.

### 4.7 Fine-Tune a Standalone LLM

```bash
python finetune_llm.py \
  --input-zip ./datasets/training_corpus.zip \
  --base-model llama3:8b-instruct-q4_K_M \
  --output-dir ./finetuned-model
```

### 4.8 Evaluate Mentor Feedback Quality

```bash
python evaluate_mentor_feedback.py \
  --input-dir ./filtered/json \
  --output-json ./reports/mentor_feedback_scores.json
```

### 4.9 Inspect Submissions Locally (GUI)

```bash
python submission_viewer_conversation.py \
  --sqlite-path /path/to/mathforum.db
```

For the web version (with optional mentor meta-feedback collection):

```bash
python submission_viewer_conversation_web.py \
  --sqlite-path /path/to/mathforum.db \
  --host 0.0.0.0 --port 8000
```

## 5. Troubleshooting Tips

- Use `OLLAMA_HOST=localhost:<port>` to target different Ollama instances when running parallel jobs.
- If you encounter GPU driver issues, verify `nvidia-smi` works and reinstall the NVIDIA driver (`sudo apt install -y nvidia-driver-580-open`).
- For PostgreSQL exporters, confirm network access to the database and that SSL settings match your deployment.
- Always re-run `pip install -r requirements.txt` after pulling repository updates to capture new dependencies.

## 6. Next Steps

- Automate data processing with `make` or shell scripts tailored to your environment.
- Explore `mathforum_rag_webchat_dueling.py` for A/B testing different RAG configurations.
- Integrate evaluation outputs from `evaluate_mentor_feedback.py` into dashboards or reports for educational stakeholders.

