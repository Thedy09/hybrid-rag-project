# Retrieval comparison

Ten questions over the lab corpus. A hit means a retrieved chunk contains the gold substring.
Vector, BM25, and hybrid use the top 10. Re-ranking scores that fused shortlist and keeps 5.
Recall@5 for re-ranking is therefore 5 out of the re-ranked list. MRR uses the same lists.

## Summary

| Method | Recall@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| Vector-only | 0.80 | 1.00 | 0.883 |
| BM25-only | 0.80 | 1.00 | 0.870 |
| Hybrid (RRF) | 0.70 | 1.00 | 0.850 |
| Hybrid + re-rank | 1.00 | 1.00 | 1.000 |

## Rank of the gold chunk

| ID | Question | Vector | BM25 | Hybrid | Re-rank |
| --- | --- | ---: | ---: | ---: | ---: |
| q1 | What are the main limitations of traditional RAG? | 1 | 1 | 1 | 1 |
| q2 | How does hybrid search improve retrieval? | 1 | 1 | 1 | 1 |
| q3 | What is the difference between dense and sparse retrieval? | 1 | 1 | 1 | 1 |
| q4 | What does error code ERR-VEC-404 mean? | 1 | 1 | 1 | 1 |
| q5 | What retention period does policy POL-RET-90 require? | 1 | 1 | 1 | 1 |
| q6 | Which reciprocal rank fusion constant does the Northwind retrieval standard recommend? | 1 | 1 | 1 | 1 |
| q7 | Which embedding model does the Aurora Search Appliance model ASA-210 ship with? | 3 | 2 | 2 | 1 |
| q8 | How can a searcher that understands synonyms still fail to surface a product code? | 1 | 5 | 2 | 1 |
| q9 | Which cross-encoder does the lab standard use to rerank fused chunks? | 2 | 1 | 2 | 1 |
| q10 | What collection rebuild service target is named by code SLA-15M? | 1 | 1 | 1 | 1 |

## Where the methods disagree

- q7: hybrid rank 2 beat vector rank 3 on “Which embedding model does the Aurora Search Appliance model ASA-210 ship with?”.
- q7: re-ranking moved the gold chunk from hybrid rank 2 to 1.
- q8: vector rank 1 beat hybrid rank 2 on “How can a searcher that understands synonyms still fail to surface a product code?”.
- q8: re-ranking moved the gold chunk from hybrid rank 2 to 1.
- q9: BM25 rank 1 beat vector rank 2, and hybrid landed at 2.
- q9: re-ranking moved the gold chunk from hybrid rank 2 to 1.
