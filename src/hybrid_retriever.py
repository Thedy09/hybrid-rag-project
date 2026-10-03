"""Combine dense and sparse ranked lists with reciprocal rank fusion."""

from __future__ import annotations

from src.settings import RRF_CONSTANT


def reciprocal_rank_fusion(result_lists: list[list[dict]], constant: int = RRF_CONSTANT) -> list[dict]:
    """Fuse ranked lists without requiring their scores to share a scale.

    A chunk at rank ``r`` (starting at 0) contributes ``1 / (constant + r + 1)``.
    With the lab constant of 60, the first hit in a list contributes 1/61.
    Chunks that appear in both lists accumulate both contributions.
    """

    fused_scores: dict[str, float] = {}
    document_map: dict[str, object] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            document = result["document"]
            document_id = document.metadata.get(
                "chunk_id",
                document.page_content[:100],
            )
            document_map[document_id] = document
            fused_scores[document_id] = fused_scores.get(document_id, 0) + 1 / (
                constant + rank + 1
            )

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    return [
        {
            "document": document_map[document_id],
            "score": fused_scores[document_id],
        }
        for document_id in ranked_ids
    ]


class HybridRetriever:
    """Run dense and sparse search, then fuse the two lists."""

    def __init__(self, dense, sparse, rrf_constant: int = RRF_CONSTANT) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_constant = rrf_constant

    def search(
        self,
        query: str,
        top_k: int = 10,
        dense_k: int = 10,
        sparse_k: int = 10,
        metadata_filter: dict | None = None,
    ) -> dict:
        dense_results = self.dense.search(
            query,
            top_k=dense_k,
            metadata_filter=metadata_filter,
        )
        sparse_results = self.sparse.search(
            query,
            top_k=sparse_k,
            metadata_filter=metadata_filter,
        )
        fused_results = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            constant=self.rrf_constant,
        )[:top_k]
        return {
            "dense": dense_results,
            "sparse": sparse_results,
            "fused": fused_results,
        }
