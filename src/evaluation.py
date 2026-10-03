"""Compare vector, BM25, hybrid, and re-ranked retrieval on the lab questions."""

from __future__ import annotations

import json
from pathlib import Path

from src.hybrid_retriever import reciprocal_rank_fusion
from src.paths import COMPARISON_PATH, QUESTIONS_PATH, RESULTS_PATH


def gold_rank(results: list[dict], needle: str) -> int | None:
    folded = needle.lower()
    for rank, result in enumerate(results, start=1):
        if folded in result["document"].page_content.lower():
            return rank
    return None


def recall_at(rank: int | None, k: int) -> int:
    return int(rank is not None and rank <= k)


def reciprocal(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def evaluate(pipeline: RAGPipeline, questions: list[dict]) -> dict:
    rows = []
    for item in questions:
        question = item["question"]
        needle = item["gold_substring"]
        dense = pipeline.dense.search(question, top_k=10)
        sparse = pipeline.sparse.search(question, top_k=10)
        fused = reciprocal_rank_fusion([dense, sparse])[:10]
        reranked = pipeline.reranker.rerank(question, fused, top_k=5)
        lists = {
            "vector": dense,
            "bm25": sparse,
            "hybrid": fused,
            "rerank": reranked,
        }
        ranks = {name: gold_rank(results, needle) for name, results in lists.items()}
        rows.append(
            {
                "id": item["id"],
                "question": question,
                "gold_substring": needle,
                "gold_file": item["gold_file"],
                "intent": item.get("intent", ""),
                "ranks": ranks,
                "tops": {name: _top_summary(results) for name, results in lists.items()},
            }
        )

    summary = {}
    for method in ("vector", "bm25", "hybrid", "rerank"):
        method_ranks = [row["ranks"][method] for row in rows]
        summary[method] = {
            "recall_at_1": sum(recall_at(rank, 1) for rank in method_ranks) / len(rows),
            "recall_at_5": sum(recall_at(rank, 5) for rank in method_ranks) / len(rows),
            "mrr": sum(reciprocal(rank) for rank in method_ranks) / len(rows),
        }
    return {"summary": summary, "questions": rows}


def write_reports(report: dict, comparison_path: Path = COMPARISON_PATH, results_path: Path = RESULTS_PATH) -> None:
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    comparison_path.write_text(_comparison_markdown(report), encoding="utf-8")


def _comparison_markdown(report: dict) -> str:
    labels = {
        "vector": "Vector-only",
        "bm25": "BM25-only",
        "hybrid": "Hybrid (RRF)",
        "rerank": "Hybrid + re-rank",
    }
    lines = [
        "# Retrieval comparison",
        "",
        "Ten questions over the lab corpus. A hit means a retrieved chunk contains the gold substring.",
        "Vector, BM25, and hybrid use the top 10. Re-ranking scores that fused shortlist and keeps 5.",
        "Recall@5 for re-ranking is therefore 5 out of the re-ranked list. MRR uses the same lists.",
        "",
        "## Summary",
        "",
        "| Method | Recall@1 | Recall@5 | MRR |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, label in labels.items():
        metrics = report["summary"][method]
        lines.append(
            f"| {label} | {metrics['recall_at_1']:.2f} | {metrics['recall_at_5']:.2f} | {metrics['mrr']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Rank of the gold chunk",
            "",
            "| ID | Question | Vector | BM25 | Hybrid | Re-rank |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["questions"]:
        ranks = row["ranks"]
        lines.append(
            "| {id} | {question} | {vector} | {bm25} | {hybrid} | {rerank} |".format(
                id=row["id"],
                question=row["question"].replace("|", "\\|"),
                vector=_fmt_rank(ranks["vector"]),
                bm25=_fmt_rank(ranks["bm25"]),
                hybrid=_fmt_rank(ranks["hybrid"]),
                rerank=_fmt_rank(ranks["rerank"]),
            )
        )

    lines.extend(["", "## Where the methods disagree", ""])
    notes = _disagreement_notes(report["questions"])
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("All four methods placed the gold chunk at the same rank on every question.")
    lines.append("")
    return "\n".join(lines)


def _disagreement_notes(rows: list[dict]) -> list[str]:
    notes = []
    for row in rows:
        ranks = row["ranks"]
        vector_rank = ranks["vector"]
        hybrid_rank = ranks["hybrid"]
        bm25_rank = ranks["bm25"]
        rerank_rank = ranks["rerank"]
        if _better(hybrid_rank, vector_rank):
            notes.append(
                f"{row['id']}: hybrid rank {_fmt_rank(hybrid_rank)} beat vector rank {_fmt_rank(vector_rank)} "
                f"on “{row['question']}”."
            )
        elif _better(vector_rank, hybrid_rank):
            notes.append(
                f"{row['id']}: vector rank {_fmt_rank(vector_rank)} beat hybrid rank {_fmt_rank(hybrid_rank)} "
                f"on “{row['question']}”."
            )
        if _better(bm25_rank, vector_rank) and not _better(hybrid_rank, vector_rank):
            notes.append(
                f"{row['id']}: BM25 rank {_fmt_rank(bm25_rank)} beat vector rank {_fmt_rank(vector_rank)}, "
                f"and hybrid landed at {_fmt_rank(hybrid_rank)}."
            )
        if _better(rerank_rank, hybrid_rank):
            notes.append(
                f"{row['id']}: re-ranking moved the gold chunk from hybrid rank {_fmt_rank(hybrid_rank)} "
                f"to {_fmt_rank(rerank_rank)}."
            )
    return notes


def _better(left: int | None, right: int | None) -> bool:
    if left is None:
        return False
    if right is None:
        return True
    return left < right


def _fmt_rank(rank: int | None) -> str:
    return "—" if rank is None else str(rank)


def _top_summary(results: list[dict]) -> dict | None:
    if not results:
        return None
    document = results[0]["document"]
    return {
        "file": document.metadata.get("file_name"),
        "page": document.metadata.get("page_number"),
        "chunk_id": document.metadata.get("chunk_id"),
        "score": results[0]["score"],
        "snippet": document.page_content[:180].replace("\n", " "),
    }


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))
