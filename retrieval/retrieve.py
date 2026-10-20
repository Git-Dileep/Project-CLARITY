import os
import json
import logging
from typing import Dict, Any, List
import chromadb
from chromadb.utils import embedding_functions

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "retrieval.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "clarity_kb"


def get_collection():
    """Load the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def extract_query_from_alert(alert: Dict[str, Any]) -> str:
    """Extract a search query from an alert adhering to alert_schema_v1."""
    parts = []
    if "attack_type" in alert:
        parts.append(str(alert["attack_type"]))
    if "predicted_label" in alert:
        parts.append(str(alert["predicted_label"]))
    if "top_features" in alert and isinstance(alert["top_features"], list):
        feat_names = [f.get("feature", "") for f in alert["top_features"] if isinstance(f, dict)]
        parts.extend(feat_names)
    if "description" in alert:
        parts.append(alert["description"])

    return " ".join([p for p in parts if p]).strip() or "generic network attack"


def retrieve_for_alert(alert: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    """
    Takes an alert, queries ChromaDB, logs the event, and returns top-k chunks with scores.
    """
    collection = get_collection()
    query_text = extract_query_from_alert(alert)
    alert_id = alert.get("alert_id", "unknown_alert")

    results = collection.query(query_texts=[query_text], n_results=top_k)

    retrieved_chunks = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        ids = results["ids"][0] if "ids" in results else ["unknown"] * len(docs)
        metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
        distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

        for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            retrieved_chunks.append({
                "chunk_id": chunk_id,
                "text": doc_text,
                "distance": float(dist),
                "similarity_score": round(1.0 - float(dist), 4),
                "metadata": meta
            })

    output = {
        "alert_id": alert_id,
        "query_used": query_text,
        "top_k": top_k,
        "chunks": retrieved_chunks
    }

    # Log retrieval call
    logging.info(f"Retrieved {len(retrieved_chunks)} chunks for alert_id={alert_id} query='{query_text}'")

    return output
