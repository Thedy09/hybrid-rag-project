"""Lab defaults shared by indexing, retrieval, and the Streamlit app."""

from __future__ import annotations

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "hybrid_rag"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

RRF_CONSTANT = 60
DENSE_K = 10
SPARSE_K = 10
FUSED_K = 10
RERANK_K = 5
