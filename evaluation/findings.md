# When hybrid search outperformed vector search

The full pipeline (dense + BM25 + reciprocal rank fusion + cross-encoder) beat vector-only retrieval on this corpus. Fusion by itself did not.

| Method | Recall@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| Vector-only | 0.80 | 1.00 | 0.883 |
| BM25-only | 0.80 | 1.00 | 0.870 |
| Hybrid (RRF) | 0.70 | 1.00 | 0.850 |
| Hybrid + re-rank | 1.00 | 1.00 | 1.000 |

Recall@5 is 1.00 for every method. The gold chunk was always inside the top 10. The differences are about order.

## Where fusion beat the vector list

**ASA-210 embedding checkpoint (q7).** “Which embedding model does the Aurora Search Appliance model ASA-210 ship with?” Vector search returned the manual’s opening first (gold rank 3). That opening says “search appliance” and “error identifiers” and does not name the checkpoint. BM25 ranked the real sentence, `ASA-210 ships with sentence-transformers/all-MiniLM-L6-v2`, at rank 2. Fusion kept rank 2, which is better than the vector rank of 3. The cross-encoder then moved that sentence to rank 1.

## Where fusion did not beat the vector list

**Synonym failure (q8).** “How can a searcher that understands synonyms still fail to surface a product code?” The explanation is in `dense-vs-sparse.md`. Vector search had it at rank 1. BM25 ranked the manual’s “record the product code” ticket instructions first, and the explanation only at rank 5. Fusion split the difference and left the explanation at rank 2. This is the usual RRF cost: a list that was already right gets pulled toward the other list.

**Cross-encoder name (q9).** BM25 had the chunk that names `cross-encoder/ms-marco-MiniLM-L-6-v2` at rank 1. Vector search had a later chunk from the same file at rank 1, and the model-name chunk at rank 2. Fusion followed the vector order and stayed at rank 2.

## Where the re-ranker finished the job

On q7, q8, and q9 the cross-encoder put the gold chunk at rank 1. Hybrid plus re-ranking is the run with Recall@1 of 1.00 and MRR of 1.000, against 0.80 and 0.883 for vector-only.

The other seven questions were already rank 1 in both indexes, including `ERR-VEC-404`, `POL-RET-90`, and `SLA-15M`. On a 46-chunk corpus those strings are distinctive enough that the embedding matches them too. A larger corpus, or a query that does not repeat the identifier, is where BM25’s advantage shows up before fusion. It already shows up inside q7 and q8: the vector list preferred a topical neighbor, and the keyword list preferred the exact string or the repeated phrase.

Re-ranking only reorders the fused top 10. It helped here because the gold chunk was always in that shortlist.
