# README.md

## Overview
This repository documents a research and development pipeline focused on integrating AI-based intelligent feedback into mathematics education, specifically within the context of the MathForum dataset and the EnCoMPASS environment. The project spans data extraction and transformation, GUI and tooling development, AI model integration, and deployment in local and cloud environments. This work supports educational research by enabling automated evaluation and feedback generation for student-submitted math problems.

## Contents
- `pow_report_with_mrubric.py`: SQLite descriptive statistics generator about the PoW Mathforum database.  This supports multiple ollama instances by invoking `OLLAMA_HOST=localhost:$PORT ollama serve &` and `OLLAMA_HOST=localhost:$PORT ollama run $MODEL` 
for the desired model used with `--ollama-model` wuen running the script, *i.e.* `mistral:7b-instruct-q4_K_M` or `llama3:8b-instruct-q4_K_M`.
`OLLAMA_HOST=localhost:$PORT ollama run llama3:8b-instruct-q4_K_M` for each port.
- `textualize_conversation.py`: SQLite extractor and formatter for threaded student submissions.
- `textualize_conversation_postgres.py`: PostgreSQL version of the above for scalable data handling.
- `filter_text_problem_outputs_conversation.py`: Applies configurable rubric/text filters to extracted JSON data.
- `filter_text_problem_outputs_conversation.json`: Configuration file defining rubric/text thresholds.
- `submission_viewer_conversation.py`: Tkinter GUI to interactively inspect submissions and mentor responses.  A web version that accepts meta-feedback is provided by `submission_viewer_conversation_web.py`.
- `mathforum_txt_rag_append.py`: Vectorization and RAG setup script using LangChain, Ollama, and Chroma.
- `mathforum_rag_webchat.py`: RAG Web frontend (and `mathforum_rag_webchat_duelling.py` to show a side-by-side RAG comparison).
- `add_metadata_vectordb.py`: Given a json document and an original document from the input to the vectorstore, look up the document id and append the json document as metadata.
- `finetune_rag.py`: Fine tune the RAG chroma embedding db and the LLM from json sample documents.  This can be used with the modified `mathforum_rag_webchat_finetuned.py` program in place of the RAG Web frontend above.
- `evaluate_mentor_feedback.py`: Given a directory of text documents from the textualize/filter scripts, generate json documents that rate the quality of the mentor feedback on a rubric.
- `finetune_llm.py`: Given a zip of json documents representing the data corpus, finetune an LLM and enable chat with that finetuned LLM.

---

## 1. Pedagogical Foundations

### Intelligent Feedback for Mathematics Learning
- Each submission is accompanied by rubric ratings and textual mentor responses.
- Rubrics include: **Strategy**, **Interpretation**, **Accuracy**, **Completeness**, **Clarity**, **Reflection**.
- AI-based feedback is compared to human mentor feedback for **A/B testing** with expert educators.

### Teacher-In-The-Loop AI
- Proposed enhancements to EnCoMPASS allow AI to prepopulate the teacher "Respond" text box.
- Input: problem statement, student solution, noticing/wondering excerpts.
- Output: coherent, rubric-aligned feedback using system prompt: _"You are a mathematics teacher."_

### Assessment Research
- Enables analysis of problem-specific and cross-problem feedback patterns.
- Facilitates research into rubric reliability, student misconceptions, and improvement over time.

---

## 2. Cloud Infrastructure and System Setup

### Recommended VM Configuration
- VM Size: `Standard D8as v5` or `Standard NC8as T4 v3 (8 vcpus, 56 GiB memory)` (currently available in the South Central US region)
- RAM: `64 GB`
- Disk: `512 GB Premium SSD`
- OS: `Ubuntu 22.04 LTS`
- **Disable Secure Boot**
- Enable port 7860 for use with webchat
- Default user: `azureuser`
- Use ssh key login
- Disable TPM and SecureBoot security features to enable CUDA

### SSH Configuration

```bash
sudo nano /etc/ssh/sshd_config
```

Set `PasswordAuthentication no`

```bash
sudo systemctl restart ssh
```

### Enable Swap

```bash
sudo swapoff /swapfile
sudo rm /swapfile

sudo fallocate -l 64GB /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapon --show
```

### Nvidia Driver Installation

```bash
sudo apt-get purge 'nvidia-*'
sudo apt-get autoremove --purge
sudo apt-get clean
sudo apt-get autoclean
sudo apt update
sudo apt install -y nvidia-driver-580-open # or 575, etc.
# reboot
nvidia-smi
```

### Python Installation

```bash
sudo apt install -y python3 python3-pip python3-venv build-essential python3-tk python-is-python3 
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo '. ~/venv/bin/activate' >> ~/.bashrc
source ~/.bashrc
```

### CUDA Installation

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y build-essential dkms
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update

wget http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb
sudo dpkg -i libtinfo5_6.3-2ubuntu0.1_amd64.deb

sudo apt install -y linux-headers-$(uname -r)

sudo apt --purge remove '^nvidia' '^cuda' '^libcuda' 'cuda*'
sudo apt autoremove

curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub

echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda.list

sudo apt update
sudo apt install -y cuda

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Pip Package Requirements
```bash
python -m pip install langchain chromadb tiktoken unstructured langchain-huggingface langchain-ollama langchain-chroma transformers sentence-transformers openai torch torchvision torchaudio tqdm gradio tk asyncpg aiohttp flask
```

---

## 3. AI Technical Stack and Workflow

### Step 1: Textualize Conversations
```bash
python textualize_conversation.py  # or textualize_conversation_postgres.py
```

### Step 2: Filter with Rubrics
Edit `filter_text_problem_outputs_conversation.json` with desired filters.

```bash
python filter_text_problem_outputs_conversation.py
```

### Step 3: Import into RAG Pipeline
Assumes you have a `data.zip` file populated with txt files.  This can be the original zip from `textualize_conversation.py` or filtered via `filter_text_problem_outputs_conversation.py`.

```bash
echo 'ulimit -n 65535' >> ~/.bashrc
echo 'export SBCL_DYNAMIC_SPACE_SIZE=32768' >> ~/.bashrc
source ~/.bashrc
python mathforum_txt_rag_append.py
```

#### Launch Ollama Model
Perform CUDA installation below first to utilize CUDA.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3
ollama serve
```

#### Launch Webchat

Open port `7860` under port rules and run:

```bash
python mathforum_rag_webchat.py
```

---

## 4. Submission Viewer Tool (Tkinter GUI)
Log into the remote machine with `-Y` to enable X forwarding.

```bash
python submission_viewer_conversation.py
```

---

## 5. PostgreSQL and SQLite Migration
The default password used by the scripts is `postgres`.

```bash
sudo apt install -y postgresql postgresql-contrib pgloader

sudo -i -u postgres
psql
\password postgres
\q
exit

createdb -U postgres -h localhost -p 5432 mathforum

export SBCL_DYNAMIC_SPACE_SIZE=32768
ulimit -n 65535

pgloader --with "prefetch rows = 1000" sqlite://mathforum.db postgresql://postgres:postgres@localhost:5432/mathforum
```

---

## 6. Using `tmux`
```bash
sudo apt install -y tmux

tmux new -s rag_session
ulimit -n 65535
python mathforum_txt_rag_append.py

# Detach
Ctrl + B, then D

# Re-attach
tmux attach -t rag_session
```

---

## 7. Resources and References

### LLM + RAG
- https://github.com/ollama/ollama
- https://docs.llamaindex.ai

### Education Research
- EnCoMPASS
- Gates Foundation AI Pedagogy

---

## 8. Future Work
- Triplet RAG (problem + submission + feedback)
- LLM fine-tuning on rubric criteria
- Streamlit GUI for educator annotation
- Multi-agent rubric modeling
- GPT-4 vs LLaMA feedback comparison

---

## 9. Script Reference

- **`textualize_conversation.py`**  
  Extracts student submissions and mentor responses from the SQLite `mathforum.db`, cleans HTML and control characters, and writes out per-thread `.txt` and `.json` files containing the full threaded conversation – now including `mentor_name` pulled from the DB.

- **`textualize_conversation_postgres.py`**  
  An async/parallelized version of the above that uses PostgreSQL (`asyncpg` + `ThreadPoolExecutor`) for large-scale exports. Builds a global student metadata map, streams submissions, and writes `.txt`/`.json` outputs with `mentor_name` included.

- **`filter_text_problem_outputs.py`**  
  Reads a ZIP of raw JSON/TXT outputs, applies configurable filters (via `filter_text_problem_outputs.json`), and writes:
  1. A filtered ZIP of JSON+TXT  
  2. An Excel workbook (`.xlsx`) with sheets for the main filter and per-rubric subsets, using `xlsxwriter` :contentReference.

- **`filter_text_problem_outputs_conversation.py`**  
  Similar to the above, but:
  - Operates on conversation-structured JSON  
  - Flattens each thread into “wide” CSV rows  
  - Adds word-level diffs between successive `short_answer`, `long_answer`, and `response` fields (via `difflib.ndiff`)  
  - Computes rubric deltas (`+N`/`-N`) per submission  
  - Outputs:
    - Filtered ZIP of JSON+TXT  
    - Two CSVs: main filter and full conversation details  
    - Per-rubric CSVs and ZIPs :contentReference.

- **`submission_viewer_conversation.py`**  
  A Tkinter GUI that:
  - Loads processed submissions from SQLite  
  - Sanitizes HTML, decodes base64 images into `[Placeholder]` tokens  
  - Displays a table with filters (ID, puzzle text, score, etc.)  
  - Opens detail windows showing full text, inline diffs, rubric scores, and clickable image previews :contentReference.

- **`mathforum_rag_webchat.py`**
  RAG Web frontend that opens a port accessible via the web.
  
- **`add_metadata_vectordb.py`** 
  Given a json document and an original document from the input to the vectorstore, look up the document id and append the json document as metadata.  This assumes the document hasn't been modified, because the document id is defined as the hash of the contents of the input file to the vectordb.  This is useful to add human expert feedback about the document.
  
  
## Filter Strategy

The filtering criteria applied to the main dataset is as follows.  It ensures that a short answer of at least 5 characters, a long answer of at least 10 characters, and a mentor response of at least 15 characters is given.  It also ensures that rubric values of at least 1 are present for all criteria, and that the short/long answer doesn't contain the string `student looked at puzzle without submitting`.  It also ensures that each rubric score improved by at least 1.  Similarly, to ensure a mentor response is present, mentor responses of `(no reply yet)` are filtered out (these are default values inserted during the textualize process.  The `contains` and `not_contains` filters will keep threads if any message contains any of the target strings in the `contains` key, and reject threads if all messages contain any of the target strings in the `not_contains` key.  The filtering configuration file is given below:

```json
{
  "short_answer": {
    "min_length": 5,
    "not_contains": "student looked at puzzle without submitting" 
  },
  "long_answer": {
    "min_length": 10,
    "not_contains": "student looked at puzzle without submitting" 
  },
  "response": {
    "min_length": 15,
    "not_contains":  ["(no reply yet)", "Write your reply here"]  
  },
  "strategy": {
    "min_value": 1,
    "is_rubric": true,
    "improvement": 1
  },
  "interpretation": {
    "min_value": 1,
    "is_rubric": true,
    "improvement": 1
  },
  "clarity": {
    "min_value": 1,
    "is_rubric": true,
    "improvement": 1
  },
  "completeness": {
    "min_value": 1,
    "is_rubric": true,
    "improvement": 1
  },
  "reflection": {
    "min_value": 1,
    "is_rubric": true,
    "improvement": 1
  },
  "accuracy": {
    "min_value": 1,
    "is_rubric": true,
    "improvement": 1
  },
  "mentor_replies": {
    "min_value": 2,
    "is_synthetic": true
  }    
}
```

--

## Acknowledgements
Developed by William Mongan and collaborators under the EnCoMPASS project, using the MathForum corpus to support equitable, automated formative assessment in mathematics.
