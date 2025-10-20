# ollama run llama3
# ollama serve

# rag_webui.py

import os
import argparse
import gradio as gr
from pathlib import Path
from tqdm import tqdm
from chromadb import PersistentClient
import langchain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
# from langchain_community.retrievers import EnsembleRetriever
from langchain_core.callbacks import StdOutCallbackHandler
import torch

# === CLI Argument Parsing ===
def parse_args():
    parser = argparse.ArgumentParser(description="RAG Web UI for Math Forum")
    parser.add_argument("--vectorstore", type=str, default="vectorstore", 
                       help="Path to vectorstore directory")
    parser.add_argument("--docs", type=str, default=None,
                       help="Path to documents directory (for reference)")
    parser.add_argument("--ollama-model", type=str, default="llama3",
                       help="Ollama model name")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port for Gradio interface")
    parser.add_argument("--top-k", type=int, default=4,
                       help="Number of documents to retrieve")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="LLM temperature")
    parser.add_argument("--debug", action="store_true",
                       help="Enable LangChain debug mode")
    return parser.parse_args()

args = parse_args()

# === Configuration ===
VECTORSTORE_DIR = args.vectorstore
COLLECTION_NAME = "vectordb"
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
OLLAMA_MODEL_NAME = args.ollama_model
OLLAMA_TEMPERATURE = args.temperature
RETRIEVER_TOP_K = args.top_k
langchain.debug = args.debug

# === Prompt Definitions ===

BASE_PROMPT = "You are a helpful mathematics master teacher who specializes in coaching new teachers to give better feedback to their students."

# === Editable System Prompt ===
SYSTEM_MESSAGE_TEMPLATE = (
    f"{BASE_PROMPT}\n\n"
    "Use the embedded documents and the context below to answer questions clearly and precisely."
)

# ChatPromptTemplate (used in 'stuff' chain)
system_message = SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE_TEMPLATE)
human_message = HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}")
chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

# PromptTemplates (used in 'refine' chain)
QUESTION_PROMPT_TEMPLATE = (
    f"{BASE_PROMPT}\n\n"
    "Use the following context to answer the question at the end.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {question}\n"
    "Answer:"
)

REFINE_PROMPT_TEMPLATE = (
    "The original question is: {question}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We also have some additional context below:\n"
    "{context_str}\n"
    "Given the new context, refine the original answer if needed. "
    "If the context isn't useful, return the original answer."
)

question_prompt = PromptTemplate(
    template=QUESTION_PROMPT_TEMPLATE,
    input_variables=["context_str", "question"]
)

refine_prompt = PromptTemplate(
    template=REFINE_PROMPT_TEMPLATE,
    input_variables=["question", "existing_answer", "context_str"]
)

# === Load Embedding Model ===
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})

# === Load All Chroma Batches with tqdm and error handling ===
retrievers = []
batch_dirs = sorted(Path(VECTORSTORE_DIR).glob("batch_*"))

if not batch_dirs:
    raise ValueError(f"No Chroma batches found in {VECTORSTORE_DIR}")

print(f"Loading Chroma batches from: {VECTORSTORE_DIR}")
for batch_dir in tqdm(batch_dirs, desc="Loading vectorstore batches"):
    try:
        collection_name = f"{COLLECTION_NAME}_{batch_dir.name}"
        client = PersistentClient(path=str(batch_dir))
        vectordb = Chroma(
            persist_directory=str(batch_dir),
            collection_name=collection_name,
            client=client,
            embedding_function=embedding
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
        retrievers.append(retriever)
    except Exception as e:
        print(f"Error loading {batch_dir}: {e}")

if not retrievers:
    raise RuntimeError("No valid retrievers could be created from the Chroma batches.")

# === Simple Retriever (using first available) ===
ensemble_retriever = retrievers[0] if retrievers else None

# === LLM Setup ===
callbacks = [StdOutCallbackHandler()]
ollama_base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
llm = ChatOllama(model=OLLAMA_MODEL_NAME, base_url=ollama_base, callbacks=callbacks, temperature=OLLAMA_TEMPERATURE)

# === RAG Chain ===
# Define a simple formatter for documents
def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Use the existing chat_prompt for consistency
prompt = chat_prompt

# Compose the LCEL graph
rag_inputs = RunnableParallel(
    context = ensemble_retriever | _format_docs,
    question = RunnablePassthrough()
)
qa = rag_inputs | prompt | llm

# === Gradio Query Function ===
def query_rag_system(user_query):
    formatted_query = f"query: {user_query}" if EMBEDDING_MODEL_NAME.startswith("intfloat/e5-") else user_query
    print("==== User Query ====")
    print(formatted_query)

    try:
        # Manually invoke retrieval for logging
        retrieved_docs = ensemble_retriever.invoke(formatted_query)
        print("==== Retrieved Documents ====")
        for i, doc in enumerate(retrieved_docs):
            print(f"[Doc {i+1} - metadata: {doc.metadata}]:\n{doc.page_content[:500]}\n")

        # Now run the full chain
        result = qa.invoke(formatted_query)

        print("==== Final Answer ====")
        answer = result.content if hasattr(result, "content") else str(result)
        print(answer)
        return answer

    except Exception as e:
        print("==== Error ====")
        print(str(e))
        return f"Error: {str(e)}"

# === Gradio Interface ===
interface = gr.Interface(
    fn=query_rag_system,
    inputs=gr.Textbox(lines=4, placeholder="Ask your question here..."),
    outputs="text",
    title="RAG Web UI",
    description="Ask questions about your Chroma-embedded documents using a local LLM."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=args.port)
