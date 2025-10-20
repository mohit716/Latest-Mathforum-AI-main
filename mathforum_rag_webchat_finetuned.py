# rag_webui_finetuned.py
# Gradio UI using fine-tuned embedding and LLM models
# pip install -U langchain-community
# pip install -U langchain langchain-community langchain-core
# pip install langchain-chroma chromadb

import os
import gradio as gr
from pathlib import Path
from tqdm import tqdm
from chromadb import PersistentClient
import langchain
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_core.callbacks import StdOutCallbackHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
import torch

# === Configuration ===
VECTORSTORE_DIR = "vectorstore_reembedded"
COLLECTION_NAME = "vectordb"
EMBEDDING_MODEL_PATH = "./fine_tuned_embeddings"
LLM_MODEL_PATH = "./fine_tuned_llm"
RETRIEVER_TOP_K = 10
OLLAMA_TEMPERATURE = 0.1
CHAIN_TYPE = "stuff"
langchain.debug = True

# === Prompt Definitions ===
BASE_PROMPT = "You are a helpful mathematics master teacher who specializes in coaching new teachers to give better feedback to their students."

SYSTEM_MESSAGE_TEMPLATE = (
    f"{BASE_PROMPT}\n\n"
    "Use the embedded documents and the context below to answer questions clearly and precisely."
)

system_message = SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE_TEMPLATE)
human_message = HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}")
chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

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

# === Load Fine-Tuned Embedding Model ===
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
embedding = HuggingFaceEmbeddings(model=embedding_model)

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

ensemble_retriever = EnsembleRetriever(
    retrievers=retrievers,
    weights=[1.0] * len(retrievers)
)

# === Load Fine-Tuned LLM ===
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")

gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=OLLAMA_TEMPERATURE,
    do_sample=True,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=gen_pipeline)

# === RAG Chain ===
chain_kwargs = {}
if CHAIN_TYPE == "refine":
    chain_kwargs = {
        "question_prompt": question_prompt,
        "refine_prompt": refine_prompt
    }
elif CHAIN_TYPE == "stuff":
    chain_kwargs = {
        "prompt": chat_prompt
    }
else:
    raise ValueError(f"Unsupported CHAIN_TYPE: {CHAIN_TYPE}")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=ensemble_retriever,
    chain_type=CHAIN_TYPE,
    chain_type_kwargs=chain_kwargs,
    input_key="question"
)

# === Gradio Query Function ===
def query_rag_system(user_query):
    formatted_query = f"query: {user_query}" if EMBEDDING_MODEL_PATH.endswith("e5-large-v2") else user_query
    print("==== User Query ====")
    print(formatted_query)

    try:
        retrieved_docs = ensemble_retriever.invoke(formatted_query)
        print("==== Retrieved Documents ====")
        for i, doc in enumerate(retrieved_docs):
            print(f"[Doc {i+1} - metadata: {doc.metadata}]:\n{doc.page_content[:500]}\n")

        result = qa_chain.invoke({"question": formatted_query})

        print("==== Final Answer ====")
        print(result["result"])
        return result["result"]

    except Exception as e:
        print("==== Error ====")
        print(str(e))
        return f"Error: {str(e)}"

# === Gradio Interface ===
interface = gr.Interface(
    fn=query_rag_system,
    inputs=gr.Textbox(lines=4, placeholder="Ask your question here..."),
    outputs="text",
    title="RAG Web UI (Fine-Tuned)",
    description="Ask questions about your Chroma-embedded documents using a fine-tuned local LLM and embeddings."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
