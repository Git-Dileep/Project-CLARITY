# Retrieval Module

This module handles dense vector embeddings, vector database persistence, and live verification caching.

## Responsibilities
- Local ChromaDB vector database index setup
- MITRE ATT&CK and IoT threat corpus chunking and embeddings ($k=5$)
- Tavily Search API live query caching
- Emitting retrieved context chunks conforming strictly to `contracts/retrieval_schema_v1.json`
