# ollama run llama3
# ollama serve

import os
import gradio as gr
from pathlib import Path
from tqdm import tqdm
from chromadb import PersistentClient
import langchain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.runnables import RunnableLambda
import torch

# === Configuration ===
VECTORSTORE_DIR_1 = "vectorstore"  
VECTORSTORE_DIR_2 = "vectorstore"  
COLLECTION_NAME = "vectordb"
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
OLLAMA_MODEL_NAME = "llama3"
OLLAMA_TEMPERATURE = 0.1
langchain.debug = True

# Per-vectorstore configuration
RETRIEVER_TOP_K_1 = 10
RETRIEVER_TOP_K_2 = 10
CHAIN_TYPE_1 = "none" # stuff, refine, none (for no RAG) ...
CHAIN_TYPE_2 = "stuff"

# === Prompt Setup ===
BASE_PROMPT = "You are a helpful mathematics master teacher who specializes in coaching new teachers to give better feedback to their students."

SYSTEM_MESSAGE_TEMPLATE = (
    f"{BASE_PROMPT}\n\n"
    "Use the embedded documents and the context below to answer questions clearly and precisely."
)

system_message = SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE_TEMPLATE)
human_message = HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}")
chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

question_prompt = PromptTemplate(
    template=(
        f"{BASE_PROMPT}\n\n"
        "Use the following context to answer the question at the end.\n\n"
        "Context:\n{context_str}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
    input_variables=["context_str", "question"]
)

refine_prompt = PromptTemplate(
    template=(
        "The original question is: {question}\n"
        "We have provided an existing answer: {existing_answer}\n"
        "We also have some additional context below:\n"
        "{context_str}\n"
        "Instructions:\n"
        "- If the new context adds nothing useful or relevant, return the existing answer **exactly as-is**, with no rewording.\n"
        "- If the new context adds value or clarification, modify the existing answer accordingly.\n"
        "- Do not make stylistic or semantic changes unless necessary based on the new context.\n\n"
    ),
    input_variables=["question", "existing_answer", "context_str"]
)

# === Embeddings ===
embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# === LLM ===
callbacks = [StdOutCallbackHandler()]
llm = ChatOllama(model=OLLAMA_MODEL_NAME, callbacks=callbacks, temperature=OLLAMA_TEMPERATURE)

# === Load Vectorstores ===
def load_ensemble_retriever(vectorstore_dir, collection_prefix, top_k):
    retrievers = []
    batch_dirs = sorted(Path(vectorstore_dir).glob("batch_*"))

    if not batch_dirs:
        raise ValueError(f"No Chroma batches found in {vectorstore_dir}")

    print(f"Loading batches from: {vectorstore_dir}")
    for batch_dir in tqdm(batch_dirs, desc=f"Loading batches from {vectorstore_dir}"):
        try:
            collection_name = f"{collection_prefix}_{batch_dir.name}"
            client = PersistentClient(path=str(batch_dir))
            vectordb = Chroma(
                persist_directory=str(batch_dir),
                collection_name=collection_name,
                client=client,
                embedding_function=embedding
            )
            retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
            retrievers.append(retriever)
        except Exception as e:
            print(f"Error loading {batch_dir}: {e}")

    if not retrievers:
        raise RuntimeError(f"No valid retrievers found for {vectorstore_dir}")

    return EnsembleRetriever(retrievers=retrievers, weights=[1.0] * len(retrievers))
    
retriever1 = None
retriever2 = None

if CHAIN_TYPE_1 != "none":
    retriever1 = load_ensemble_retriever(VECTORSTORE_DIR_1, COLLECTION_NAME, RETRIEVER_TOP_K_1)

if CHAIN_TYPE_2 != "none":
    retriever2 = load_ensemble_retriever(VECTORSTORE_DIR_2, COLLECTION_NAME, RETRIEVER_TOP_K_2)

# === Chains ===
def build_chain(chain_type, retriever=None):
    if chain_type == "none":
        # Plain LLM-only chain, no retrieval
        def llm_only_chain(inputs):
            question = inputs["question"] if isinstance(inputs, dict) else inputs
            prompt = f"{BASE_PROMPT}\n\nQuestion: {question}\nAnswer:"
            response = llm.invoke(prompt)
            return {"result": response.content}
        return RunnableLambda(llm_only_chain)

    if chain_type == "refine":
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="refine",
            chain_type_kwargs={
                "question_prompt": question_prompt,
                "refine_prompt": refine_prompt
            },
            input_key="question"
        )

    elif chain_type == "stuff":
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": chat_prompt},
            input_key="question"
        )

    else:
        raise ValueError(f"Unsupported CHAIN_TYPE: {chain_type}")

qa_chain1 = build_chain(CHAIN_TYPE_1, retriever1)
qa_chain2 = build_chain(CHAIN_TYPE_2, retriever2)

# === Gradio Query Function ===
def query_dual_rag_system(user_query):
    formatted_query = f"query: {user_query}" if EMBEDDING_MODEL_NAME.startswith("intfloat/e5-") else user_query

    try:
        if retriever1 is not None:
            print("==== Vectorstore 1 Retrieval ====")
            docs1 = retriever1.invoke(formatted_query)
            for i, doc in enumerate(docs1):
                print(f"[VS1 Doc {i+1}] {doc.page_content[:300]}")
        else:
            print("==== Vectorstore 1: No RAG (LLM only) ====")

        if retriever2 is not None:
            print("==== Vectorstore 2 Retrieval ====")
            docs2 = retriever2.invoke(formatted_query)
            for i, doc in enumerate(docs2):
                print(f"[VS2 Doc {i+1}] {doc.page_content[:300]}")
        else:
            print("==== Vectorstore 2: No RAG (LLM only) ====")

        print("==== Vectorstore 1: Invoke ====")
        result1 = qa_chain1.invoke({"question": formatted_query})
        print("==== Vectorstore 2: Invoke ====")
        result2 = qa_chain2.invoke({"question": formatted_query})

        print("==== Results ====")
        print("VS1:", result1["result"])
        print("VS2:", result2["result"])

        return result1["result"], result2["result"]

    except Exception as e:
        error_message = f"Error: {str(e)}"
        return error_message, error_message

# === Gradio UI ===
with gr.Blocks(title="RAG Web UI") as interface:
    gr.Markdown("## RAG Web UI")
    gr.Markdown("Submit your question.")

    with gr.Row():
        question_box = gr.Textbox(lines=4, placeholder="Ask your question here...", label="Your Question")
        
    submit_button = gr.Button("Submit")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Vectorstore 1")
            output1 = gr.Textbox(label="", lines=10)
        with gr.Column():
            gr.Markdown("### Vectorstore 2")
            output2 = gr.Textbox(label="", lines=10)

    submit_button.click(fn=query_dual_rag_system, inputs=question_box, outputs=[output1, output2])

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
