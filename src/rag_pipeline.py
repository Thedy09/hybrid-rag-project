"""End-to-end hybrid retrieval and grounded answer generation."""

from __future__ import annotations

import time
from pathlib import Path

from dotenv import load_dotenv

from src.bootstrap import configure_environment

configure_environment()

from src.dense_retriever import DenseRetriever
from src.generator import Generator
from src.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from src.ingestion import load_chunks, load_documents, save_chunks, split_documents
from src.paths import CHUNKS_PATH, DATA_DIR, ROOT, VECTOR_DIR
from src.reranker import Reranker
from src.settings import DENSE_K, FUSED_K, RERANK_K, SPARSE_K
from src.sparse_retriever import BM25Retriever

PROMPT_TEMPLATE = """You are a question-answering assistant.

Use only the supplied context to answer the question.

Requirements:
- Do not invent information.
- If the answer is unavailable, say that the context does not contain it.
- Provide the source file and page number.
- Keep the answer concise and factual.

Context:
{context}

Question:
{question}
"""


class RAGPipeline:
    """Dense search, BM25, reciprocal rank fusion, re-ranking, then an answer."""

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        reranker: Reranker,
        generator: Generator,
        chunks: list,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.hybrid = HybridRetriever(dense, sparse)
        self.reranker = reranker
        self.generator = generator
        self.chunks = chunks

    @classmethod
    def build(
        cls,
        data_directory: str | Path = DATA_DIR,
        persist_directory: str | Path = VECTOR_DIR,
        chunks_path: str | Path = CHUNKS_PATH,
    ) -> RAGPipeline:
        load_dotenv(ROOT / ".env")
        documents = load_documents(data_directory)
        if not documents:
            raise FileNotFoundError(f"No PDF, text, or Markdown files found in {data_directory}")
        chunks = split_documents(documents)
        save_chunks(chunks, chunks_path)
        dense = DenseRetriever(persist_directory=str(persist_directory))
        dense.create_vector_store(chunks)
        return cls._assemble(dense, chunks)

    @classmethod
    def load(
        cls,
        data_directory: str | Path = DATA_DIR,
        persist_directory: str | Path = VECTOR_DIR,
        chunks_path: str | Path = CHUNKS_PATH,
        force: bool = False,
    ) -> RAGPipeline:
        load_dotenv(ROOT / ".env")
        chunks_file = Path(chunks_path)
        vector_dir = Path(persist_directory)
        stale = _corpus_is_stale(data_directory, chunks_file)
        ready = chunks_file.exists() and vector_dir.exists() and any(vector_dir.iterdir())
        if force or stale or not ready:
            return cls.build(data_directory, persist_directory, chunks_path)

        chunks = load_chunks(chunks_file)
        dense = DenseRetriever(persist_directory=str(persist_directory))
        dense.load_vector_store()
        return cls._assemble(dense, chunks)

    @classmethod
    def _assemble(cls, dense: DenseRetriever, chunks: list) -> RAGPipeline:
        return cls(
            dense=dense,
            sparse=BM25Retriever(chunks),
            reranker=Reranker(),
            generator=Generator(),
            chunks=chunks,
        )

    def answer(
        self,
        question: str,
        metadata_filter: dict | None = None,
        dense_k: int = DENSE_K,
        sparse_k: int = SPARSE_K,
        fused_k: int = FUSED_K,
        rerank_k: int = RERANK_K,
    ) -> dict:
        started = time.perf_counter()
        timings: dict[str, float] = {}

        mark = time.perf_counter()
        dense_results = self.dense.search(
            question,
            top_k=dense_k,
            metadata_filter=metadata_filter,
        )
        timings["dense_ms"] = _elapsed_ms(mark)

        mark = time.perf_counter()
        sparse_results = self.sparse.search(
            question,
            top_k=sparse_k,
            metadata_filter=metadata_filter,
        )
        timings["sparse_ms"] = _elapsed_ms(mark)

        mark = time.perf_counter()
        fused_results = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            constant=self.hybrid.rrf_constant,
        )[:fused_k]
        timings["fusion_ms"] = _elapsed_ms(mark)

        mark = time.perf_counter()
        reranked_results = self.reranker.rerank(question, fused_results, top_k=rerank_k)
        timings["rerank_ms"] = _elapsed_ms(mark)

        context = format_context(reranked_results)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        mark = time.perf_counter()
        generation = self.generator.generate(prompt, question, reranked_results)
        timings["generate_ms"] = _elapsed_ms(mark)

        return {
            "question": question,
            "answer": generation.answer,
            "generator": generation.backend,
            "generator_warning": generation.warning,
            "prompt": prompt,
            "context": context,
            "dense_results": dense_results,
            "sparse_results": sparse_results,
            "fused_results": fused_results,
            "reranked_results": reranked_results,
            "latency_ms": _elapsed_ms(started),
            "timings_ms": timings,
            "metadata_filter": metadata_filter or {},
        }


def format_context(results: list[dict]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        document = result["document"]
        file_name = document.metadata.get("file_name", "unknown")
        page = document.metadata.get("page_number", "?")
        blocks.append(
            f"[Source {index}: {file_name}, Page {page}]\n{document.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _corpus_is_stale(data_directory: str | Path, chunks_path: Path) -> bool:
    if not chunks_path.exists():
        return True
    root = Path(data_directory)
    if not root.exists():
        return True
    newest = max(
        (path.stat().st_mtime for path in root.iterdir() if path.is_file()),
        default=0,
    )
    return newest > chunks_path.stat().st_mtime
