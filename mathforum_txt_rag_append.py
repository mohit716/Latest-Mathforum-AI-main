"""
Offline RAG setup and querying using LangChain from a ZIP of text files.

---------------------------------------------------------------
Recommended Azure VM:
    - VM Size: Standard D4as v5 or D8as v5
    - Memory: 64 GB RAM
    - Disk: 512 GB Premium SSD
    - OS: Ubuntu 22.04 LTS
    - Rationale: Optimal for parallel txt parsing, embedding, and Chroma vector storage
---------------------------------------------------------------

What it does:
1. Extracts a ZIP file of txt documents, skipping 0-byte files.
2. Applies configurable filters (e.g., value thresholds, string lengths).
3. Removes specified keys to reduce embedding size.
4. Loads filtered documents into LangChain Documents.
5. Creates a local Chroma vector store with HuggingFace embeddings.
6. If the vector store exists, skips setup and uses it directly.
7. Launches an interactive query loop using LangChain and Ollama.

---------------------------------------------------------------
Required pip installs:
    pip install langchain chromadb tiktoken unstructured langchain-huggingface langchain-ollama langchain-chroma
    pip install transformers sentence-transformers
    pip install openai  # optional, for compatibility with some chains
    pip install tqdm
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
---------------------------------------------------------------

---------------------------------------------------------------
To configure global constants:

FILTERS = [
    {"key": "score", "type": "value", "op": "<", "threshold": 5},
    {"key": "description", "type": "length", "op": ">", "threshold": 20},
]

REMOVE_KEYS = ["debug_info", "raw_html", "internal_notes"]

ARCHIVE_PATH = "data.zip"  # Or data.tar.bz2
EXTRACT_DIR = "unzipped_data"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # sentence-transformers/all-MiniLM-L6-v2, sentence-transformers/all-mpnet-base-v2, intfloat/e5-large-v2, or others, but note that some changes to the pre-processing, embedding, and querying may be required for each model, which should be reflected in the writer and query loops
OLLAMA_MODEL_NAME = "llama3"
BATCH_SIZE = 50
NUM_WORKERS = multiprocessing.cpu_count() + 2
SPLITTER_CHUNK_SIZE = 512
SPLITTER_CHUNK_OVERLAP = 50
CHUNK_BATCH_LIMIT = 5000 
INGEST_BATCH_LIMIT = 5000 # max for Chroma is approx. 5000
EMBEDDING_BATCH_SIZE = 64
COLLECTION_NAME = 'vectordb'
RETRIEVER_TOP_K = 10
MAX_QUEUE_SIZE = 100
MAX_FILES = None  # Set to an integer (e.g., 100) to limit for testing
---------------------------------------------------------------

To run ollama:

sudo snap install ollama
ollama pull llama3
ollama serve
"""

############################################################################################################

import os
import zipfile
import gc
import tarfile
import multiprocessing
from pathlib import Path
from typing import List
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import torch
from math import ceil
from uuid import uuid4
from multiprocessing import Process, Queue, cpu_count
from queue import Empty as QueueEmpty
import atexit
import time
import threading
import asyncio
from hashlib import sha256
from pydantic import Field

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_core.runnables import Runnable
from langchain.retrievers import EnsembleRetriever

os.environ["TOKENIZERS_PARALLELISM"] = "true"

ARCHIVE_PATH = "data.zip"
EXTRACT_DIR = "unzipped_data"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
OLLAMA_MODEL_NAME = "llama3"
BATCH_SIZE = 2000
SPLITTER_CHUNK_SIZE = 512
SPLITTER_CHUNK_OVERLAP = 50
CHUNK_BATCH_LIMIT = 1000
INGEST_BATCH_LIMIT = 5000 # max for Chroma is approx. 5000
EMBEDDING_BATCH_SIZE = 256
COLLECTION_NAME = 'vectordb'
RETRIEVER_TOP_K = 10
MAX_QUEUE_SIZE = 500
MAX_FILES = None
NUM_WORKERS = multiprocessing.cpu_count()
FILTERS = []
REMOVE_KEYS = []

STOP_SIGNAL = "<STOP>"

def prompt_yes_no(prompt_str: str) -> bool:
    while True:
        user_input = input(f"{prompt_str} [y/n]: ").strip().lower()
        if user_input in {"y", "yes"}:
            return True
        elif user_input in {"n", "no"}:
            return False
        else:
            print("Please enter 'y' or 'n'.")
            
def load_existing_doc_ids_and_filter_files_by_source(
    vectorstore_dir: str,
    collection_name_prefix: str,
    files: List[str]
) -> tuple[set, List[str]]:
    """
    - Loads all existing doc_ids from the Chroma vectorstore.
    - Maps them to their source file via metadata["source"].
    - Filters out input files for which *all* associated doc_ids already exist.
    
    Returns:
        - Set of all existing doc_ids
        - List of files that should be processed (i.e., have at least one missing doc_id)
    """
    existing_doc_ids = set()
    source_to_doc_ids = {}

    batch_dirs = sorted(Path(vectorstore_dir).glob("batch_*"))
    for batch_dir in tqdm(batch_dirs, desc="Loading Chroma collections", leave=False):
        collection_name = f"{collection_name_prefix}_{batch_dir.name}"
        client = PersistentClient(path=str(batch_dir))
        collection = client.get_or_create_collection(collection_name)

        offset = 0
        while True:
            results = collection.get(include=["metadatas"], offset=offset, limit=BATCH_SIZE)
            if not results["metadatas"]:
                break
            for metadata in results["metadatas"]:
                doc_id = metadata.get("doc_id")
                source = metadata.get("source")
                if doc_id and source:
                    existing_doc_ids.add(doc_id)
                    source_to_doc_ids.setdefault(source, set()).add(doc_id)
            offset += BATCH_SIZE

    # Filter out files whose all doc_ids already exist
    files_to_process = []
    for filepath in tqdm(files, desc="Filtering files", leave=False):
        path_str = str(filepath)
        file_hash = sha256(Path(filepath).read_bytes()).hexdigest()
        if (
            file_hash not in existing_doc_ids
            or path_str not in source_to_doc_ids
            or len(source_to_doc_ids[path_str]) == 0
        ):
            files_to_process.append(filepath)

    return existing_doc_ids, files_to_process
    
def log_worker(progress_queue, bars):
    import time
    start_times = {}
    while True:
        try:
            msg = progress_queue.get()
            
            if msg == STOP_SIGNAL:
                break

            bar_name, action, value = msg

            if bar_name not in bars:
                bars[bar_name] = tqdm(
                    total=0,
                    desc=bar_name,
                    dynamic_ncols=True,
                    unit="item",
                    smoothing=0.2,
                )
                start_times[bar_name] = time.time()

            bar = bars[bar_name]

            if action == "add_total":
                bar.total += value
                bar.refresh()

            elif action == "update":
                bar.update(value)
                elapsed = time.time() - start_times[bar_name]
                if elapsed > 0:
                    rate = bar.n / elapsed
                    bar.set_postfix(rate=f"{rate:.2f} it/s")

            elif action == "postfix":
                bar.set_postfix(**value)
        except Exception as e:
            tqdm.write(f"[Warning] Failed process log worker command: {e}")
            continue

def cleanup_cuda():
    torch.cuda.empty_cache()
    
def cleanup_memory():
    gc.collect()

def extract_archive_skipping_0_bytes(archive_path: str, extract_to: str) -> List[str]:
    os.makedirs(extract_to, exist_ok=True)
    kept_files = []
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            for zip_info in tqdm(zip_ref.infolist(), desc="Extracting ZIP", leave=False):
                if zip_info.file_size > 0:
                    zip_ref.extract(zip_info, extract_to)
                    extracted_path = os.path.join(extract_to, zip_info.filename)
                    if extracted_path.endswith(".txt"):
                        kept_files.append(extracted_path)
    elif archive_path.endswith(('.tar.bz2', '.tar.gz', '.tar')):
        mode = "r:bz2" if archive_path.endswith(".bz2") else "r:gz" if archive_path.endswith(".gz") else "r:"
        with tarfile.open(archive_path, mode) as tar:
            for member in tqdm(tar.getmembers(), desc="Extracting TAR", leave=False):
                if member.isfile() and member.size > 0:
                    tar.extract(member, extract_to)
                    extracted_path = os.path.join(extract_to, member.name)
                    if extracted_path.endswith(".txt"):
                        kept_files.append(extracted_path)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    return kept_files

def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def process_txt_file(filepath: str, filters: List[dict], remove_keys_list: List[str]) -> Document:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text_content = f.read()

        doc_id = sha256(Path(filepath).read_bytes()).hexdigest()

        return Document(
            page_content=text_content,
            metadata={
                "source": str(filepath),
                "doc_id": doc_id,
            }
        )
    except Exception as e:
        tqdm.write(f"[Error] Failed to load TXT from {filepath}: {e}")
        return None

def process_file_star(args):
    return process_txt_file(*args)
        
def prefix(blob, prefix_text="passage: "):
    if isinstance(blob, str):
        return f"{prefix_text}{blob}"
    elif isinstance(blob, list):
        for chunk in blob:
            if hasattr(chunk, "page_content"):
                chunk.page_content = f"{prefix_text}{chunk.page_content}"
        return blob
    else:
        raise TypeError("Expected string or list of documents with page_content.")

def embedding_process_fn(chunk_queue, embedding_queue, stop_signal, embedding_model, progress_queue):
    embedding = SentenceTransformer(embedding_model, device="cuda" if torch.cuda.is_available() else "cpu")

    while True:
        try:
            item = chunk_queue.get(timeout=1)
        except QueueEmpty:
            continue

        if item == stop_signal:
            embedding_queue.put(stop_signal)
            break
            
        texts, metadatas, ids = item
        text_batches = list(batched(list(zip(texts, ids)), EMBEDDING_BATCH_SIZE))

        progress_queue.put(("Embedding", "add_total", len(texts)))
        embeddings = []
        id_out = []

        for text_id_batch in text_batches:
            text_batch, id_batch = zip(*text_id_batch)
            embedded = embedding.encode(text_batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.extend(embedded)
            id_out.extend(id_batch)
            
            progress_queue.put(("Embedding", "update", len(text_batch)))
            progress_queue.put(("Embedding", "postfix", {"chunk_q": chunk_queue.qsize(), "progress_q": progress_queue.qsize()}))

        embedding_queue.put((texts, metadatas, embeddings, id_out))

        del item, texts, metadatas, embeddings, ids, text_batches, id_out
        cleanup_cuda()
        cleanup_memory()

def ingestion_process_fn(embedding_queue, stop_signal, vectorstore_dir, collection_name, progress_queue):
    client = PersistentClient(path=vectorstore_dir)
    collection = client.get_or_create_collection(collection_name)

    buffer = []
    buffer_doc_count = 0
    buffer_max_docs = INGEST_BATCH_LIMIT

    def flush_buffer():
        nonlocal buffer_doc_count
        nonlocal buffer
                
        if not buffer:
            return

        all_texts, all_metadatas, all_embeddings, all_ids = [], [], [], []
        
        for texts, metadatas, embeddings, ids in buffer:
            all_texts.extend(texts)
            all_metadatas.extend(metadatas)
            all_embeddings.extend(embeddings)
            all_ids.extend(ids)

        collection.add(
            documents=all_texts,
            metadatas=all_metadatas,
            embeddings=all_embeddings,
            ids=all_ids
        )
                
        progress_queue.put(("Ingesting into Chroma", "add_total", len(all_texts)))
        progress_queue.put(("Ingesting into Chroma", "update", len(all_texts)))
        progress_queue.put(("Ingesting into Chroma", "postfix", {
            "embed_q": embedding_queue.qsize(),
            "progress_q": progress_queue.qsize(),
        }))
        
        buffer.clear()
        buffer_doc_count = 0
        cleanup_memory()

    while True:
        try:
            item = embedding_queue.get(timeout=2)
            
            if item == stop_signal:
                flush_buffer()
                break

            num_docs = len(item[0])  # item[0] is the list of texts
            if buffer_doc_count + num_docs > buffer_max_docs:
                flush_buffer()

            buffer.append(item)
            buffer_doc_count += num_docs
            
            if embedding_queue.empty():
                flush_buffer()
        except QueueEmpty:
            flush_buffer()

def process_files_parallel(filepaths, filters, remove_keys, num_workers=None, progress_queue=None, existing_doc_ids=None):
    documents = []
    num_workers = num_workers or multiprocessing.cpu_count()
    args_iterable = [(fp, filters, remove_keys) for fp in filepaths]

    if progress_queue:
        progress_queue.put(("Processing files", "add_total", len(filepaths)))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for doc in executor.map(process_file_star, args_iterable):
            if doc is not None:
                doc_id = doc.metadata.get("doc_id")
                if not existing_doc_ids or doc_id not in existing_doc_ids:
                    documents.append(doc)
            if progress_queue:
                progress_queue.put(("Processing files", "update", 1))

    return documents

def split_documents_parallel(documents, num_workers=None, progress_queue=None):
    num_workers = num_workers or multiprocessing.cpu_count()
    splitter = RecursiveCharacterTextSplitter(chunk_size=SPLITTER_CHUNK_SIZE, chunk_overlap=SPLITTER_CHUNK_OVERLAP)
    chunks = []
    futures = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(splitter.split_documents, [doc]) for doc in documents]
        if progress_queue:
            progress_queue.put(("Splitting documents", "add_total", len(futures)))
        for future in as_completed(futures):
            chunks.extend(future.result())
            if progress_queue:
                progress_queue.put(("Splitting documents", "update", 1))
                
    return chunks

def create_vectorstore_streaming_mp(filepaths, batch_index, batch_size=100, progress_queue=None, log_bars=None, existing_doc_ids=None):
    batch_dir = os.path.join(VECTORSTORE_DIR, f"batch_{batch_index}")
    os.makedirs(batch_dir, exist_ok=True)
    collection_name = f"{COLLECTION_NAME}_batch_{batch_index}"

    chunk_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    embedding_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)

    embed_proc = multiprocessing.Process(
        target=embedding_process_fn,
        args=(chunk_queue, embedding_queue, STOP_SIGNAL, EMBEDDING_MODEL_NAME, progress_queue)
    )
    ingest_proc = multiprocessing.Process(
        target=ingestion_process_fn,
        args=(embedding_queue, STOP_SIGNAL, batch_dir, collection_name, progress_queue)
    )

    embed_proc.daemon = True
    ingest_proc.daemon = True

    embed_proc.start()
    ingest_proc.start()

    documents = process_files_parallel(filepaths, FILTERS, REMOVE_KEYS, NUM_WORKERS, progress_queue=progress_queue, existing_doc_ids=existing_doc_ids)
    chunks = split_documents_parallel(documents, NUM_WORKERS, progress_queue=progress_queue)

    if EMBEDDING_MODEL_NAME.startswith("intfloat/e5-"):
        chunks = prefix(chunks, prefix_text="passage: ")

    if chunks:
        progress_queue.put(("Extracting embedding batches", "add_total", len(chunks)))
        for chunk_batch in batched(chunks, CHUNK_BATCH_LIMIT):
            texts = [chunk.page_content for chunk in chunk_batch]
            metadatas = [chunk.metadata for chunk in chunk_batch]
            ids = [str(uuid4()) for _ in chunk_batch]
            chunk_queue.put((texts, metadatas, ids))
            progress_queue.put(("Extracting embedding batches", "update", len(chunk_batch)))
        cleanup_memory()

    chunk_queue.put(STOP_SIGNAL)
    embed_proc.join()
    ingest_proc.join()

    client = PersistentClient(path=batch_dir)
    vectordb = Chroma(persist_directory=batch_dir, client=client, collection_name=collection_name)
    collection = client.get_collection(collection_name)

    return batch_dir, vectordb, collection

def setup_rag_from_zip():
    vector_dbs = []
    vectorstore_exists = os.path.exists(VECTORSTORE_DIR) and any(Path(VECTORSTORE_DIR).glob("batch_*"))
    extracted_exists = os.path.exists(EXTRACT_DIR) and os.listdir(EXTRACT_DIR)
    archive_exists = os.path.exists(ARCHIVE_PATH)
    archive_or_extracted_exists = archive_exists or extracted_exists

    if extracted_exists:
        files = list(Path(EXTRACT_DIR).rglob("*.txt"))
    elif archive_exists:
        files = extract_archive_skipping_0_bytes(ARCHIVE_PATH, EXTRACT_DIR)
    else:
        files = []

    existing_doc_ids = set()
    files_to_process = files

    if vectorstore_exists and archive_or_extracted_exists:
        tqdm.write("Vectorstore and data archive/unzipped data directory detected.")
        if prompt_yes_no("Would you like to scan for new TXT files and add any missing documents to the existing vectorstore (set ulimit -n high)?"):
            existing_doc_ids, files_to_process = load_existing_doc_ids_and_filter_files_by_source(
                VECTORSTORE_DIR,
                COLLECTION_NAME,
                files
            )
        else:
            files_to_process = []

    if MAX_FILES is not None:
        files_to_process = files_to_process[:MAX_FILES]

    batch_dirs = sorted(Path(VECTORSTORE_DIR).glob("batch_*"))
    existing_batch_indices = []
    for bdir in batch_dirs:
        try:
            idx = int(bdir.name.replace("batch_", ""))
            existing_batch_indices.append(idx)
        except ValueError:
            continue
    start_batch_index = max(existing_batch_indices, default=-1) + 1

    progress_queue = multiprocessing.Queue()
    log_bars = {}
    log_thread = threading.Thread(target=log_worker, args=(progress_queue, log_bars), daemon=True)
    log_thread.start()

    if files_to_process:
        tqdm.write(f"Scanning {len(files_to_process)} files and skipping already ingested ones...")
        total_batches = (len(files_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
        progress_queue.put(("Vectorstore Batches", "add_total", total_batches))

        for offset, batch_files in enumerate(batched(files_to_process, BATCH_SIZE)):
            batch_index = start_batch_index + offset
            progress_queue.put(("Vectorstore Batches", "update", 1))
            batch_dir, vectordb, collection = create_vectorstore_streaming_mp(
                batch_files,
                batch_index=batch_index,
                progress_queue=progress_queue,
                log_bars=log_bars,
                existing_doc_ids=existing_doc_ids
            )
            vector_dbs.append((str(batch_dir), vectordb, collection))

    progress_queue.put(STOP_SIGNAL)
    log_thread.join()

    all_batch_dirs = sorted(Path(VECTORSTORE_DIR).glob("batch_*"))
    existing_vector_dbs = []
    for batch_dir in all_batch_dirs:
        collection_name = f"{COLLECTION_NAME}_{batch_dir.name}"
        client = PersistentClient(path=str(batch_dir))
        vectordb = Chroma(persist_directory=str(batch_dir), client=client, collection_name=collection_name)
        collection = client.get_collection(collection_name)
        existing_vector_dbs.append((str(batch_dir), vectordb, collection))

    all_dirs = {batch_dir: (vectordb, collection) for batch_dir, vectordb, collection in vector_dbs + existing_vector_dbs}
    return [
    {
        "persist_directory": batch_dir,
        "collection_name": collection.name,
        "client": client
    }
    for batch_dir, collection in all_dirs.values()
    ]


def start_query_loop(vector_dbs):
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    retrievers = [
        Chroma(
            persist_directory=info["persist_directory"],
            collection_name=info["collection_name"],
            client=info["client"],
            embedding_function=embedding
        ).as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
        for info in vector_dbs
    ]

    retriever = EnsembleRetriever(retrievers=retrievers, weights=[1.0] * len(retrievers))

    llm = ChatOllama(model=OLLAMA_MODEL_NAME)

    system_message = SystemMessagePromptTemplate.from_template(
        "You are a helpful mathematics master teacher who specializes in coaching new teachers to give better feedback to their students. Answer questions using the embedded documents and the provided context.")
    human_message = HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": chat_prompt},
        input_key="question"
    )

    while True:
        query = input("Ask a question (or type 'exit'): ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if EMBEDDING_MODEL_NAME.startswith("intfloat/e5-"):
            query = prefix(query, prefix_text="query: ")

        try:
            response = qa_chain.invoke({"question": query})
            tqdm.write(f"\nAnswer: {response}\n")
        except Exception as e:
            tqdm.write(f"Error during query: {e}\n")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    vector_dbs = setup_rag_from_zip()
    if vector_dbs:
        start_query_loop(vector_dbs)