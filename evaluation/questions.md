# Evaluation questions

These ten questions were written before the index was built. `evaluation/questions.json` is the file the scoring script reads. A hit is a retrieved chunk that contains the gold substring.

1. What are the main limitations of traditional RAG?
   Gold file: `rag-limitations.md`. Expect both indexes to find the conceptual answer.
2. How does hybrid search improve retrieval?
   Gold file: `hybrid-search.md`. Conceptual wording that matches the source.
3. What is the difference between dense and sparse retrieval?
   Gold file: `dense-vs-sparse.md`. Conceptual contrast.
4. What does error code ERR-VEC-404 mean?
   Gold file: `aurora-search-manual.txt`. Exact identifier. A topical paragraph about a missing collection does not contain the code.
5. What retention period does policy POL-RET-90 require?
   Gold file: `northwind-retrieval-policy.md`. Exact policy identifier. The general records section discusses retention without the identifier.
6. Which reciprocal rank fusion constant does the Northwind retrieval standard recommend?
   Gold file: `reciprocal-rank-fusion.md`. The named setting is RRF-K60.
7. Which embedding model does the Aurora Search Appliance model ASA-210 ship with?
   Gold file: `aurora-search-manual.txt`. Exact model and appliance identifiers. The chunking note discusses the kind of question without naming the checkpoint.
8. How can a searcher that understands synonyms still fail to surface a product code?
   Gold file: `dense-vs-sparse.md`. Paraphrase. The appliance manual repeats "product code" as a ticket instruction and does not explain this failure.
9. Which cross-encoder does the lab standard use to rerank fused chunks?
   Gold file: `cross-encoder-reranking.md`. Exact model name.
10. What collection rebuild service target is named by code SLA-15M?
    Gold file: `vector-error-codes.pdf`, page 2. The manual discusses rebuild timing and deliberately omits the code.
