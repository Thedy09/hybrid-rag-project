"""Run the ten-question retrieval comparison."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import configure_environment

configure_environment()

from src.evaluation import evaluate, load_questions, write_reports
from src.paths import COMPARISON_PATH, RESULTS_PATH
from src.rag_pipeline import RAGPipeline


def main() -> None:
    pipeline = RAGPipeline.load()
    report = evaluate(pipeline, load_questions())
    write_reports(report)
    print(f"Wrote {COMPARISON_PATH}")
    print(f"Wrote {RESULTS_PATH}")
    for method, metrics in report["summary"].items():
        print(
            f"{method:8}  R@1={metrics['recall_at_1']:.2f}  "
            f"R@5={metrics['recall_at_5']:.2f}  MRR={metrics['mrr']:.3f}"
        )


if __name__ == "__main__":
    main()
