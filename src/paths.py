"""Project paths resolved from this file so commands work from any cwd."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
VECTOR_DIR = ROOT / "vector_store"
ARTIFACTS_DIR = ROOT / "artifacts"
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.jsonl"
EVAL_DIR = ROOT / "evaluation"
QUESTIONS_PATH = EVAL_DIR / "questions.json"
COMPARISON_PATH = EVAL_DIR / "comparison.md"
RESULTS_PATH = EVAL_DIR / "results.json"
