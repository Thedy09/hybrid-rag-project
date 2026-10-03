"""Keyword retrieval with BM25Okapi."""

from __future__ import annotations

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.filters import matches_metadata
from src.tokenize import tokenize


class BM25Retriever:
    """In-memory sparse index over the same chunks stored in Chroma.

    BM25 tends to win when the query contains an exact product name, error
    code, technical abbreviation, identifier, or other rare term. The tokenizer
    keeps hyphenated identifiers intact.
    """

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.tokenized_documents = [tokenize(document.page_content) for document in documents]
        self.index = BM25Okapi(self.tokenized_documents) if self.tokenized_documents else None

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        if self.index is None or top_k <= 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                break
            document = self.documents[index]
            if not matches_metadata(document.metadata, metadata_filter):
                continue
            results.append({"document": document, "score": score})
            if len(results) >= top_k:
                break
        return results
