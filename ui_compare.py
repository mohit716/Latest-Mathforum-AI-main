#!/usr/bin/env python3
"""
3-Model Comparison UI for MathForum AI
Compares outputs from: Vanilla LLM, RAG Model, and Fine-tuned Model
"""

import os
import json
import requests
import time
import gradio as gr
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import logging

# Setup logging directory
os.makedirs('logs', exist_ok=True)
os.makedirs('exports', exist_ok=True)

# Configure logging with file handler
log_filename = f'logs/ui_comparison_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Usage statistics
usage_stats = {
    'total_queries': 0,
    'model_a_success': 0,
    'model_b_success': 0,
    'model_c_success': 0,
    'model_a_errors': 0,
    'model_b_errors': 0,
    'model_c_errors': 0,
    'response_times_a': [],
    'response_times_b': [],
    'response_times_c': []
}

# Configuration from environment variables
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "dolphin-phi:latest")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./vectorstore")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FINETUNED_OLLAMA_MODEL = os.getenv("FINETUNED_OLLAMA_MODEL", "")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))

# Global variables for caching
_retriever = None
_embeddings = None

def ping_ollama() -> bool:
    """Check if Ollama is reachable"""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.ok
    except Exception as e:
        logger.warning(f"Ollama ping failed: {e}")
        return False

def make_llm(model_name: str) -> ChatOllama:
    """Create a ChatOllama instance"""
    return ChatOllama(
        base_url=OLLAMA_URL, 
        model=model_name, 
        temperature=0.2,
        timeout=30.0
    )

def build_retriever():
    """Build the RAG retriever from ChromaDB"""
    global _retriever, _embeddings
    
    if _retriever is not None:
        return _retriever
    
    try:
        logger.info(f"Loading embeddings from {EMBED_MODEL}")
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        
        logger.info(f"Loading ChromaDB from {CHROMA_DIR}")
        vs = Chroma(
            persist_directory=CHROMA_DIR, 
            embedding_function=_embeddings
        )
        
        _retriever = vs.as_retriever(search_kwargs={"k": 4})
        logger.info("RAG retriever built successfully")
        return _retriever
        
    except Exception as e:
        logger.error(f"Failed to build retriever: {e}")
        raise

def format_docs(docs):
    """Format retrieved documents for the RAG chain"""
    return "\n\n".join(doc.page_content for doc in docs)

# --- A) Vanilla LLM
def llm_answer(prompt: str) -> str:
    """Get answer from vanilla LLM"""
    try:
        logger.info(f"LLM query: {prompt[:50]}...")
        llm = make_llm(LLM_MODEL)
        response = llm.invoke(prompt)
        result = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"LLM response length: {len(result)}")
        return result
    except Exception as e:
        error_msg = f"[LLM ERROR] {str(e)}"
        logger.error(error_msg)
        return error_msg

# --- B) RAG Model
def rag_answer(prompt: str) -> str:
    """Get answer from RAG model"""
    try:
        logger.info(f"RAG query: {prompt[:50]}...")
        
        # Build retriever
        retriever = build_retriever()
        
        # Create RAG chain using modern LangChain approach
        llm = make_llm(LLM_MODEL)
        
        system_message = SystemMessagePromptTemplate.from_template(
            "You are a helpful mathematics teacher. Answer the question using the provided context. "
            "If the context doesn't contain relevant information, say so and provide a general answer."
        )
        human_message = HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\nQuestion: {question}"
        )
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        # Create RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | chat_prompt
            | llm
        )
        
        response = rag_chain.invoke(prompt)
        result = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"RAG response length: {len(result)}")
        return result
        
    except Exception as e:
        error_msg = f"[RAG ERROR] {str(e)}"
        logger.error(error_msg)
        return error_msg

# --- C) Fine-tuned Model (Ollama)
def finetuned_answer(prompt: str) -> str:
    """Get answer from fine-tuned model"""
    if not FINETUNED_OLLAMA_MODEL:
        return "[Fine-Tuned] Not configured. Set FINETUNED_OLLAMA_MODEL environment variable."
    
    try:
        logger.info(f"Fine-tuned query: {prompt[:50]}...")
        llm = make_llm(FINETUNED_OLLAMA_MODEL)
        response = llm.invoke(prompt)
        result = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Fine-tuned response length: {len(result)}")
        return result
    except Exception as e:
        error_msg = f"[Fine-Tuned ERROR] {str(e)}"
        logger.error(error_msg)
        return error_msg

def compare_models(prompt: str):
    """Compare all three models with timing and error tracking"""
    if not prompt.strip():
        return "Please enter a prompt!", "", ""
    
    logger.info(f"Comparing models for prompt: {prompt[:100]}...")
    usage_stats['total_queries'] += 1
    
    # Call Model A with timing
    start_time = time.time()
    try:
        llm_result = llm_answer(prompt)
        elapsed = time.time() - start_time
        usage_stats['response_times_a'].append(elapsed)
        if "[LLM ERROR]" not in llm_result:
            usage_stats['model_a_success'] += 1
            logger.info(f"Model A success ({elapsed:.2f}s)")
        else:
            usage_stats['model_a_errors'] += 1
            logger.error(f"Model A error: {llm_result}")
        llm_result = f"{llm_result}\n\n⏱️ Response Time: {elapsed:.2f}s"
    except Exception as e:
        usage_stats['model_a_errors'] += 1
        logger.error(f"Model A exception: {e}")
        llm_result = f"[Model A ERROR] {str(e)}"
    
    # Call Model B with timing
    start_time = time.time()
    try:
        rag_result = rag_answer(prompt)
        elapsed = time.time() - start_time
        usage_stats['response_times_b'].append(elapsed)
        if "[RAG ERROR]" not in rag_result:
            usage_stats['model_b_success'] += 1
            logger.info(f"Model B success ({elapsed:.2f}s)")
        else:
            usage_stats['model_b_errors'] += 1
            logger.error(f"Model B error: {rag_result}")
        rag_result = f"{rag_result}\n\n⏱️ Response Time: {elapsed:.2f}s"
    except Exception as e:
        usage_stats['model_b_errors'] += 1
        logger.error(f"Model B exception: {e}")
        rag_result = f"[Model B ERROR] {str(e)}"
    
    # Call Model C with timing
    start_time = time.time()
    try:
        finetuned_result = finetuned_answer(prompt)
        elapsed = time.time() - start_time
        usage_stats['response_times_c'].append(elapsed)
        if "[Fine-Tuned ERROR]" not in finetuned_result and "[Fine-Tuned]" not in finetuned_result:
            usage_stats['model_c_success'] += 1
            logger.info(f"Model C success ({elapsed:.2f}s)")
        else:
            usage_stats['model_c_errors'] += 1
            logger.info(f"Model C: {finetuned_result}")
        finetuned_result = f"{finetuned_result}\n\n⏱️ Response Time: {elapsed:.2f}s"
    except Exception as e:
        usage_stats['model_c_errors'] += 1
        logger.error(f"Model C exception: {e}")
        finetuned_result = f"[Model C ERROR] {str(e)}"
    
    return llm_result, rag_result, finetuned_result

def export_comparison(prompt: str, result_a: str, result_b: str, result_c: str) -> str:
    """Export comparison results to JSON file"""
    try:
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'question': prompt,
            'model_a': result_a,
            'model_b': result_b,
            'model_c': result_c,
            'usage_stats': {
                'total_queries': usage_stats['total_queries'],
                'model_a_successes': usage_stats['model_a_success'],
                'model_b_successes': usage_stats['model_b_success'],
                'model_c_successes': usage_stats['model_c_success'],
                'avg_time_a': sum(usage_stats['response_times_a']) / max(len(usage_stats['response_times_a']), 1),
                'avg_time_b': sum(usage_stats['response_times_b']) / max(len(usage_stats['response_times_b']), 1),
                'avg_time_c': sum(usage_stats['response_times_c']) / max(len(usage_stats['response_times_c']), 1)
            }
        }
        
        filename = f"exports/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported comparison to {filename}")
        return f"✅ Exported to {filename}"
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return f"❌ Export failed: {str(e)}"

def get_usage_stats() -> str:
    """Get usage statistics"""
    stats_text = f"""
# 📊 Usage Statistics

**Total Queries:** {usage_stats['total_queries']}

## Success Rates
- Model A: {usage_stats['model_a_success']} successful, {usage_stats['model_a_errors']} errors
- Model B: {usage_stats['model_b_success']} successful, {usage_stats['model_b_errors']} errors  
- Model C: {usage_stats['model_c_success']} successful, {usage_stats['model_c_errors']} errors

## Average Response Times
- Model A: {sum(usage_stats['response_times_a']) / max(len(usage_stats['response_times_a']), 1):.2f}s
- Model B: {sum(usage_stats['response_times_b']) / max(len(usage_stats['response_times_b']), 1):.2f}s
- Model C: {sum(usage_stats['response_times_c']) / max(len(usage_stats['response_times_c']), 1):.2f}s
"""
    return stats_text

def get_status_info():
    """Get status information for the UI header"""
    ollama_status = "✅ Ollama OK" if ping_ollama() else "⚠️ Ollama not reachable"
    
    chroma_status = "✅ ChromaDB OK" if os.path.exists(CHROMA_DIR) else "⚠️ ChromaDB not found"
    
    finetuned_status = f"✅ {FINETUNED_OLLAMA_MODEL}" if FINETUNED_OLLAMA_MODEL else "❌ Not configured"
    
    return f"""
    {ollama_status} • LLM: `{LLM_MODEL}` • Chroma: `{CHROMA_DIR}` • Fine-tuned: `{finetuned_status}`
    """

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # 🧠 AI Model Testing Interface (ABC Comparison)
        
        Compare responses from three different AI approaches:
        - **Model A**: Vanilla LLM (direct Ollama)
        - **Model B**: RAG Model (Retrieval-Augmented Generation)
        - **Model C**: Fine-tuned Model (if configured)
        
        {get_status_info()}
        """
    )
    
    with gr.Row():
        user_input = gr.Textbox(
            label="Enter your question", 
            placeholder="e.g., How should I give feedback on a student's algebra work?",
            lines=3
        )
    
    submit_btn = gr.Button("🔍 Compare Models", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column():
            out_a = gr.Textbox(
                label="Model A: Vanilla LLM", 
                lines=8,
                max_lines=15,
                show_copy_button=True
            )
        with gr.Column():
            out_b = gr.Textbox(
                label="Model B: RAG Model", 
                lines=8,
                max_lines=15,
                show_copy_button=True
            )
        with gr.Column():
            out_c = gr.Textbox(
                label="Model C: Fine-Tuned Model", 
                lines=8,
                max_lines=15,
                show_copy_button=True
            )
    
    # Example prompts
    gr.Markdown("### 💡 Example Questions:")
    with gr.Row():
        gr.Button("How should I give feedback on algebra work?").click(
            lambda: "How should I give feedback on a student's algebra work?", 
            outputs=user_input
        )
        gr.Button("What makes good geometry feedback?").click(
            lambda: "What makes good mentor feedback for geometry problems?", 
            outputs=user_input
        )
        gr.Button("How to improve problem-solving?").click(
            lambda: "How can I help students improve their problem-solving strategies?", 
            outputs=user_input
        )
    
    # Stats panel
    with gr.Accordion("📊 Usage Statistics", open=False):
        stats_display = gr.Markdown(get_usage_stats())
        refresh_stats_btn = gr.Button("🔄 Refresh Stats")
        refresh_stats_btn.click(
            fn=get_usage_stats,
            outputs=stats_display
        )
    
    # Export section
    with gr.Accordion("💾 Export Results", open=False):
        export_btn = gr.Button("💾 Export Last Comparison")
        export_status = gr.Textbox(label="Export Status", lines=2)
        
        def export_last(user_input, result_a, result_b, result_c):
            return export_comparison(user_input, result_a, result_b, result_c)
        
        export_btn.click(
            fn=export_last,
            inputs=[user_input, out_a, out_b, out_c],
            outputs=export_status
        )
    
    # Connect the submit button
    submit_btn.click(
        fn=compare_models, 
        inputs=user_input, 
        outputs=[out_a, out_b, out_c]
    )

if __name__ == "__main__":
    logger.info("Starting 3-Model Comparison UI...")
    logger.info(f"Ollama URL: {OLLAMA_URL}")
    logger.info(f"LLM Model: {LLM_MODEL}")
    logger.info(f"Chroma Directory: {CHROMA_DIR}")
    logger.info(f"Embedding Model: {EMBED_MODEL}")
    logger.info(f"Fine-tuned Model: {FINETUNED_OLLAMA_MODEL or 'Not configured'}")
    logger.info(f"Server Port: {SERVER_PORT}")
    
    # Test Ollama connection
    if ping_ollama():
        logger.info("✅ Ollama is reachable")
    else:
        logger.warning("⚠️ Ollama is not reachable - LLM and RAG will fail")
    
    # Test ChromaDB
    if os.path.exists(CHROMA_DIR):
        logger.info("✅ ChromaDB directory exists")
    else:
        logger.warning(f"⚠️ ChromaDB directory not found: {CHROMA_DIR}")
    
    demo.launch(
        server_name="0.0.0.0", 
        server_port=SERVER_PORT,
        show_error=True
    )


