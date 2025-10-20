# MathForum AI - Quick Start Guide

This guide provides step-by-step instructions to get the MathForum AI system running quickly with sample data.

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.10+ 
- Linux/Ubuntu environment
- Ollama installed and running
- At least 4GB RAM

### Step 1: Environment Setup
```bash
# Clone/navigate to the repository
cd /path/to/Latest-Mathforum-AI-main-1

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Verify Ollama is Running
```bash
# Check if Ollama is installed
which ollama

# List available models
ollama list

# If no models, pull one:
ollama pull dolphin-phi:latest
```

### Step 3: Run the System
```bash
# The sample data and vectorstore are already created
# Just run the web interface:
source venv/bin/activate
python mathforum_rag_webchat.py --ollama-model dolphin-phi:latest --vectorstore vectorstore --port 7860
```

### Step 4: Access the Interface

#### For Local Development:
Open your browser and go to: **http://localhost:7860**

#### For AWS/Remote Servers:
If running on AWS or a remote server, you need SSH port forwarding:

**On your local machine, run:**
```bash
ssh -L 7860:localhost:7860 username@your-server-ip
```

**Then open your browser and go to:** **http://localhost:7860**

**Alternative - Direct AWS access:**
If your AWS security group allows port 7860, you can access directly at:
**http://your-aws-public-ip:7860**

---

## 📁 Project Structure

```
Latest-Mathforum-AI-main-1/
├── venv/                          # Python virtual environment
├── sample_data/                   # Sample math problems
│   ├── problem1.txt              # Triangle area problem
│   ├── problem2.txt              # Linear equation problem
│   └── problem3.txt              # Fraction word problem
├── vectorstore/                   # ChromaDB vector database
│   └── batch_0/                  # Generated embeddings
├── data.zip                      # Sample data archive
├── mathforum_rag_webchat.py      # Main web interface
├── mathforum_txt_rag_append.py   # RAG setup script
└── requirements.txt              # Python dependencies
```

---

## 🔧 Detailed Setup Instructions

### Option A: Use Existing Sample Data (Recommended)

The repository already includes sample data and a pre-built vectorstore. Simply run:

```bash
source venv/bin/activate
python mathforum_rag_webchat.py --ollama-model dolphin-phi:latest --vectorstore vectorstore --port 7860
```

### Option B: Rebuild Vector Database

If you want to rebuild the vector database from scratch:

```bash
# 1. Ensure sample data exists
ls sample_data/  # Should show problem1.txt, problem2.txt, problem3.txt

# 2. Create data archive
zip -r data.zip sample_data/

# 3. Build vector database
source venv/bin/activate
python mathforum_txt_rag_append.py
# Note: This will ask for user input - press 'n' to skip re-scanning

# 4. Run web interface
python mathforum_rag_webchat.py --ollama-model dolphin-phi:latest --vectorstore vectorstore --port 7860
```

---

## 🎯 What You Can Do

### Sample Questions to Try:
- "How should I give feedback on a student's algebra work?"
- "What makes good mentor feedback for geometry problems?"
- "How can I help students improve their problem-solving strategies?"
- "What are effective ways to encourage mathematical reflection?"

### Features Available:
- **RAG-powered responses** based on mentor feedback examples
- **Interactive web interface** with Gradio
- **Local LLM processing** via Ollama
- **Vector similarity search** for relevant teaching strategies

---

## 🛠️ Troubleshooting

### Common Issues:

#### 1. Import Errors
```bash
# If you see LangChain import errors, ensure you're using the virtual environment:
source venv/bin/activate
pip install --upgrade langchain langchain-core langchain-community
```

#### 2. Ollama Connection Issues
```bash
# Check if Ollama is running:
ps aux | grep ollama

# Start Ollama if needed:
ollama serve &

# Verify model is available:
ollama list
```

#### 3. Port Already in Use
```bash
# Use a different port:
python mathforum_rag_webchat.py --ollama-model dolphin-phi:latest --vectorstore vectorstore --port 7861
```

#### 4. Vector Database Not Found
```bash
# Rebuild the vector database:
rm -rf vectorstore/
python mathforum_txt_rag_append.py
```

#### 5. AWS/Remote Server Access Issues
```bash
# Check if the service is running on the server:
ss -tlnp | grep 7860

# Test local access on the server:
curl http://localhost:7860

# For SSH port forwarding, use:
ssh -L 7860:localhost:7860 username@your-server-ip

# If using AWS, ensure security group allows port 7860
# Or use a different port and forward that instead
```

---

## 📊 System Requirements

### Minimum Requirements:
- **CPU**: 2+ cores
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Python**: 3.10+
- **Ollama**: Latest version

### Recommended for Best Performance:
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU with CUDA support (optional)
- **Storage**: SSD with 5GB+ free space

---

## 🔍 Understanding the Sample Data

The sample data includes three math problems with:

1. **Problem Statement**: The mathematical challenge
2. **Student Solution**: How a student approached the problem
3. **Mentor Feedback**: Expert teacher feedback with rubric scores
4. **Teaching Strategies**: Pedagogical insights and suggestions

### Rubric Categories:
- **Strategy**: Problem-solving approach
- **Interpretation**: Understanding of the problem
- **Accuracy**: Correctness of solution
- **Completeness**: Thoroughness of work
- **Clarity**: Communication of ideas
- **Reflection**: Self-assessment and learning

---

## 🚀 Advanced Usage

### Using Different Models
```bash
# Use a different Ollama model:
ollama pull llama3:8b-instruct-q4_K_M
python mathforum_rag_webchat.py --ollama-model llama3:8b-instruct-q4_K_M --vectorstore vectorstore --port 7860
```

### Customizing the Interface
```bash
# Adjust retrieval parameters:
python mathforum_rag_webchat.py --ollama-model dolphin-phi:latest --vectorstore vectorstore --port 7860 --top-k 6 --temperature 0.2
```

### Adding Your Own Data
1. Create text files with math problems and mentor feedback
2. Place them in a new directory (e.g., `my_data/`)
3. Create a zip file: `zip -r my_data.zip my_data/`
4. Update `ARCHIVE_PATH` in `mathforum_txt_rag_append.py`
5. Rebuild the vector database

---

## 📝 Next Steps

1. **Explore the Interface**: Try different questions about math teaching
2. **Add Your Data**: Include your own math problems and feedback
3. **Customize Models**: Experiment with different LLM models
4. **Extend Functionality**: Modify the code for your specific needs

---

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs**: Look for error messages in the terminal
2. **Verify dependencies**: Ensure all packages are installed correctly
3. **Test components**: Verify Ollama and the vector database separately
4. **Check resources**: Ensure sufficient RAM and disk space

---

## 📄 License & Credits

This project is part of the MathForum AI research initiative. The sample data is created for demonstration purposes and represents typical math education scenarios.

**Key Contributors**: William Mongan and collaborators under the EnCoMPASS project.

---

*Last updated: October 2024*
