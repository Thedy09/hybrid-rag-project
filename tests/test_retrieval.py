"""Tests for fusion, BM25, tokenisation, and chunk metadata. No model downloads."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.bootstrap import configure_environment

configure_environment()

from langchain_core.documents import Document

from src.evaluation import gold_rank, load_questions, recall_at
from src.hybrid_retriever import reciprocal_rank_fusion
from src.ingestion import load_documents, split_documents
from src.paths import DATA_DIR, QUESTIONS_PATH
from src.sparse_retriever import BM25Retriever
from src.tokenize import tokenize


class FakeDocument:
    def __init__(self, chunk_id: str, content: str = "x") -> None:
        self.metadata = {"chunk_id": chunk_id}
        self.page_content = content


def _hit(chunk_id: str, content: str = "x") -> dict:
    return {"document": FakeDocument(chunk_id, content), "score": 1.0}


class FusionTests(unittest.TestCase):
    def test_chunk_in_both_lists_outranks_a_single_list_winner(self) -> None:
        dense = [_hit("gold"), _hit("other")]
        sparse = [_hit("gold"), _hit("distractor")]
        fused = reciprocal_rank_fusion([dense, sparse], constant=60)
        self.assertEqual(fused[0]["document"].metadata["chunk_id"], "gold")
        self.assertAlmostEqual(fused[0]["score"], 1 / 61 + 1 / 61)

    def test_first_rank_uses_the_zero_based_formula(self) -> None:
        fused = reciprocal_rank_fusion([[_hit("only")]], constant=60)
        self.assertAlmostEqual(fused[0]["score"], 1 / 61)

    def test_missing_chunk_id_falls_back_to_a_content_prefix(self) -> None:
        prefix = "same opening text " * 6
        document = FakeDocument("ignored", prefix)
        document.metadata = {}
        other = FakeDocument("ignored-too", prefix + "extra tail")
        other.metadata = {}
        fused = reciprocal_rank_fusion(
            [[{"document": document, "score": 1}], [{"document": other, "score": 1}]]
        )
        self.assertEqual(len(fused), 1)


class BM25Tests(unittest.TestCase):
    def test_hyphenated_identifier_is_one_token(self) -> None:
        self.assertIn("err-vec-404", tokenize("What does ERR-VEC-404 mean?"))
        self.assertNotIn("err", tokenize("ERR-VEC-404"))

    def test_exact_identifier_outranks_a_topical_neighbor(self) -> None:
        documents = [
            Document(
                page_content="The retention period for customer answers is described without an identifier.",
                metadata={"chunk_id": "distractor", "file_name": "policy.md"},
            ),
            Document(
                page_content="POL-RET-90 is the policy that sets the retention period for customer answers.",
                metadata={"chunk_id": "gold", "file_name": "policy.md"},
            ),
            Document(
                page_content="Index builds run on a schedule and write a process log.",
                metadata={"chunk_id": "filler-a", "file_name": "ops.md"},
            ),
            Document(
                page_content="Disk watermarks are capacity events and not ranking results.",
                metadata={"chunk_id": "filler-b", "file_name": "ops.md"},
            ),
        ]
        results = BM25Retriever(documents).search(
            "What retention period does policy POL-RET-90 require?",
            top_k=2,
        )
        self.assertEqual(results[0]["document"].metadata["chunk_id"], "gold")

    def test_metadata_filter_drops_nonmatching_chunks(self) -> None:
        documents = [
            Document(
                page_content="POL-RET-90 is the binding retention rule.",
                metadata={"chunk_id": "policy", "file_name": "policy.md", "document_type": "md"},
            ),
            Document(
                page_content="POL-RET-90 is repeated here in a manual.",
                metadata={"chunk_id": "manual", "file_name": "manual.txt", "document_type": "txt"},
            ),
            Document(
                page_content="Index builds run on a schedule and write a process log.",
                metadata={"chunk_id": "filler-a", "file_name": "ops.md", "document_type": "md"},
            ),
            Document(
                page_content="Disk watermarks are capacity events and not ranking results.",
                metadata={"chunk_id": "filler-b", "file_name": "ops.md", "document_type": "md"},
            ),
            Document(
                page_content="Citation rules name a file and a page and nothing else.",
                metadata={"chunk_id": "filler-c", "file_name": "ops.md", "document_type": "md"},
            ),
        ]
        results = BM25Retriever(documents).search(
            "POL-RET-90",
            top_k=5,
            metadata_filter={"document_type": "txt"},
        )
        self.assertEqual([hit["document"].metadata["chunk_id"] for hit in results], ["manual"])


class IngestionTests(unittest.TestCase):
    def test_chunks_keep_file_page_type_and_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "notes.md"
            path.write_text("Sentence one stays intact. " * 80, encoding="utf-8")
            documents = load_documents(temporary)
            self.assertEqual(len(documents), 1)
            chunks = split_documents(documents)
            self.assertGreater(len(chunks), 1)
            identifiers = [chunk.metadata["chunk_id"] for chunk in chunks]
            self.assertEqual(len(identifiers), len(set(identifiers)))
            for chunk in chunks:
                self.assertEqual(chunk.metadata["file_name"], "notes.md")
                self.assertEqual(chunk.metadata["document_type"], "md")
                self.assertEqual(chunk.metadata["page_number"], 1)
                self.assertIn("notes-p1-c", chunk.metadata["chunk_id"])

    def test_gold_substrings_are_unique_to_the_expected_file(self) -> None:
        by_file: dict[str, str] = {}
        for document in load_documents(DATA_DIR):
            name = document.metadata["file_name"]
            by_file[name] = by_file.get(name, "") + "\n" + document.page_content
        for item in load_questions(QUESTIONS_PATH):
            needle = item["gold_substring"].lower()
            hits = [name for name, text in by_file.items() if needle in text.lower()]
            self.assertEqual(hits, [item["gold_file"]], item["id"])

    def test_pdf_exposes_a_real_page_number(self) -> None:
        pages = [
            document
            for document in load_documents(DATA_DIR)
            if document.metadata["file_name"] == "vector-error-codes.pdf"
        ]
        self.assertEqual(sorted(document.metadata["page_number"] for document in pages), [1, 2])
        self.assertTrue(any("SLA-15M" in document.page_content for document in pages))
        self.assertFalse(any("SLA-15M" in document.page_content and document.metadata["page_number"] == 1 for document in pages))


class MetricTests(unittest.TestCase):
    def test_gold_rank_and_recall(self) -> None:
        results = [
            {"document": FakeDocument("a", "background only")},
            {"document": FakeDocument("b", "contains SLA-15M in the procedure")},
        ]
        self.assertEqual(gold_rank(results, "sla-15m"), 2)
        self.assertEqual(recall_at(2, 1), 0)
        self.assertEqual(recall_at(2, 5), 1)
        self.assertIsNone(gold_rank(results, "missing"))


if __name__ == "__main__":
    unittest.main()
