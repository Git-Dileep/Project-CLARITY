import json
import os
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "clarity_kb"
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "kb_manifest.json")


def load_manifest(manifest_path: str = MANIFEST_PATH) -> List[Dict[str, Any]]:
    """Load the IoT knowledge base manifest."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_document(doc: Dict[str, Any], chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Split document content/summary into overlapping chunks.
    Ensures chunk size and overlap parameters remain deterministic.
    """
    full_text = f"Title: {doc['title']}. Category: {doc['category']}. Summary: {doc['summary']}"
    chunks = []
    
    start = 0
    stride = max(1, chunk_size - overlap)
    idx = 0
    
    while start < len(full_text):
        chunk_text = full_text[start : start + chunk_size]
        chunk_id = f"{doc['id']}_chk_{idx}"
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "doc_id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "source_url": doc["source_url"],
                "license": doc["license"]
            }
        })
        idx += 1
        start += stride
        if start + overlap >= len(full_text) and start < len(full_text):
            # Capture any trailing boundary
            break

    return chunks


def build_kb(manifest_path: str = MANIFEST_PATH, db_dir: str = CHROMA_PERSIST_DIR):
    """Chunk manifest entries and ingest into persistent ChromaDB."""
    os.makedirs(db_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=db_dir)
    
    # Use standard default sentence-transformers embedding
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    docs = load_manifest(manifest_path)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
        
    ids = [c["chunk_id"] for c in all_chunks]
    documents = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Ingested {len(all_chunks)} chunks from {len(docs)} documents into '{COLLECTION_NAME}'.")
    return collection


def test_retrieval(collection, queries: List[str], top_k: int = 5):
    """Run test queries to verify chunk relevance."""
    print("\n--- Running Test Queries ---")
    for q in queries:
        print(f"\nQuery: {q}")
        results = collection.query(query_texts=[q], n_results=top_k)
        for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
            print(f"  [{i+1}] Doc ID: {meta['doc_id']} | Dist: {dist:.4f}")
            print(f"      Text: {doc[:120]}...")


if __name__ == "__main__":
    col = build_kb()
    
    # 3 required test queries for acceptance criteria
    sample_queries = [
        "Mirai botnet brute force credential scanning patterns",
        "MQTT broker authentication and transport security",
        "Network denial of service UDP and SYN flood mitigation"
    ]
    test_retrieval(col, sample_queries, top_k=3)
