"""Cross-encoder re-ranking of fused chunks."""

from __future__ import annotations

from src.bootstrap import configure_environment

configure_environment()

from sentence_transformers import CrossEncoder

from src.settings import RERANKER_MODEL


class Reranker:
    """Score each query and chunk together, then keep the top chunks.

    The bi-encoder used for dense retrieval encodes the two texts separately.
    This model reads the pair at once, which is slower and more accurate, so it
    runs only on the fused shortlist.
    """

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device="cpu")

    def rerank(self, query: str, results: list[dict], top_k: int = 5) -> list[dict]:
        if not results or top_k <= 0:
            return []

        pairs = [
            [query, result["document"].page_content]
            for result in results
        ]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(results, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            {
                "document": result["document"],
                "score": float(score),
            }
            for result, score in ranked[:top_k]
        ]
