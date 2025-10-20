import sys
import os
import json
from pathlib import Path
from hashlib import sha256
from chromadb import PersistentClient

VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "vectordb"

def compute_doc_id(filepath):
    with open(filepath, "rb") as f:
        return sha256(f.read()).hexdigest()

def find_collection_for_doc(doc_id):
    for batch_dir in sorted(Path(VECTORSTORE_DIR).glob("batch_*")):
        client = PersistentClient(path=str(batch_dir))
        collection_name = f"{COLLECTION_NAME}_{batch_dir.name}"
        collection = client.get_or_create_collection(collection_name)
        try:
            results = collection.get(ids=[doc_id], include=["documents", "embeddings", "metadatas"])
            if results["ids"]:
                return collection, results
        except Exception as e:
            continue
    return None, None

def update_metadata(collection, doc_id, document, embedding, old_metadata, rubric):
    updated_metadata = old_metadata.copy()
    updated_metadata["rubric"] = rubric
    collection.update(
        ids=[doc_id],
        documents=[document],
        embeddings=[embedding],
        metadatas=[updated_metadata]
    )

def main(json_path, text_path):
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    if not os.path.exists(text_path):
        print(f"Error: Text file not found: {text_path}")
        sys.exit(1)

    with open(json_path, "r") as f:
        rubric = json.load(f)

    doc_id = compute_doc_id(text_path)
    collection, result = find_collection_for_doc(doc_id)

    if collection is None:
        print(f"Error: Document ID {doc_id} not found in any collection.")
        sys.exit(1)

    document = result["documents"][0]
    embedding = result["embeddings"][0]
    metadata = result["metadatas"][0]

    update_metadata(collection, doc_id, document, embedding, metadata, rubric)
    print(f"Metadata for document {doc_id} updated successfully with rubric.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python update_rubric.py rubric.json document.txt")
        sys.exit(1)

    json_path = sys.argv[1]
    text_path = sys.argv[2]
    main(json_path, text_path)
