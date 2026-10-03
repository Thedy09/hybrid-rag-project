"""Build the Chroma index and the BM25 chunk file."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import configure_environment

configure_environment()

from src.dense_retriever import DenseRetriever
from src.ingestion import load_documents, save_chunks, split_documents
from src.paths import CHUNKS_PATH, DATA_DIR, VECTOR_DIR
from src.sparse_retriever import BM25Retriever


def main() -> None:
    documents = load_documents(DATA_DIR)
    print(f"Loaded {len(documents)} documents")
    if not documents:
        raise SystemExit(f"No documents found in {DATA_DIR}")
    print("Sample metadata:", documents[0].metadata)

    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    print("Sample chunk metadata:", chunks[0].metadata)
    save_chunks(chunks, CHUNKS_PATH)

    dense = DenseRetriever(persist_directory=str(VECTOR_DIR))
    dense.create_vector_store(chunks)
    print(f"Vector store written to {VECTOR_DIR}")

    dense_hits = dense.search("How can retrieval accuracy be improved?", top_k=5)
    print("\nDense sample:")
    for hit in dense_hits:
        document = hit["document"]
        print(f"- {document.metadata['file_name']} similarity={hit['score']:.4f}")
        print(document.page_content[:240].replace("\n", " "))

    sparse = BM25Retriever(chunks)
    print("\nBM25 sample for ERR-VEC-404:")
    for hit in sparse.search("What does error code ERR-VEC-404 mean?", top_k=3):
        document = hit["document"]
        print(f"- {document.metadata['file_name']} bm25={hit['score']:.4f}")
        print(document.page_content[:240].replace("\n", " "))


if __name__ == "__main__":
    main()
