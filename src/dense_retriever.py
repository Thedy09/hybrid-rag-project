"""Dense retrieval with a local sentence-transformer index in Chroma."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.bootstrap import configure_environment

configure_environment()

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.filters import chroma_where, matches_metadata
from src.settings import COLLECTION_NAME, EMBEDDING_MODEL


def build_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )


def create_vector_store(
    chunks: list[Document],
    persist_directory: str = "./vector_store",
    model_name: str = EMBEDDING_MODEL,
) -> Chroma:
    """Embed chunks and persist them. An existing index at this path is replaced."""

    retriever = DenseRetriever(
        persist_directory=persist_directory,
        model_name=model_name,
    )
    return retriever.create_vector_store(chunks)


class DenseRetriever:
    """Vector search over chunk embeddings.

    Chroma returns a distance where lower is closer. Scores exposed to the
    rest of the pipeline are cosine similarities (1 - distance) so that higher
    is better, matching BM25, fusion, and the cross-encoder.
    """

    def __init__(
        self,
        persist_directory: str = "./vector_store",
        model_name: str = EMBEDDING_MODEL,
        embeddings: HuggingFaceEmbeddings | None = None,
    ) -> None:
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.embeddings = embeddings or build_embeddings(model_name)
        self.vector_store: Chroma | None = None

    def create_vector_store(self, chunks: list[Document]) -> Chroma:
        _reset_persist_directory(self.persist_directory)
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            ids=[str(chunk.metadata["chunk_id"]) for chunk in chunks],
            collection_name=COLLECTION_NAME,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )
        return self.vector_store

    def load_vector_store(self) -> Chroma:
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )
        return self.vector_store

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        if self.vector_store is None:
            self.load_vector_store()
        assert self.vector_store is not None

        where = chroma_where(metadata_filter)
        pairs = self._query(query, top_k, where)
        if metadata_filter:
            pairs = [
                pair for pair in pairs if matches_metadata(pair[0].metadata, metadata_filter)
            ]
            if len(pairs) < top_k:
                wide = self._query(query, max(top_k, 100), None)
                pairs = [
                    pair
                    for pair in wide
                    if matches_metadata(pair[0].metadata, metadata_filter)
                ]

        return [_pack_hit(document, distance) for document, distance in pairs[:top_k]]

    def _query(
        self,
        query: str,
        top_k: int,
        where: dict | None,
    ) -> list[tuple[Document, float]]:
        assert self.vector_store is not None
        kwargs = {"k": top_k}
        if where:
            kwargs["filter"] = where
        try:
            return self.vector_store.similarity_search_with_score(query, **kwargs)
        except Exception:
            if not where:
                raise
            return self.vector_store.similarity_search_with_score(query, k=max(top_k, 100))


def _pack_hit(document: Document, distance: float) -> dict:
    return {
        "document": document,
        "score": float(1.0 - distance),
        "distance": float(distance),
    }


def _reset_persist_directory(persist_directory: str) -> None:
    _clear_chroma_clients()
    path = Path(persist_directory)
    if path.exists():
        shutil.rmtree(path)


def _clear_chroma_clients() -> None:
    for module_name in (
        "chromadb.api.client",
        "chromadb.api.shared_system_client",
    ):
        try:
            module = __import__(module_name, fromlist=["SharedSystemClient"])
            clearer = getattr(module, "SharedSystemClient", None)
            if clearer is not None:
                clearer.clear_system_cache()
        except Exception:
            continue
