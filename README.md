# Hybrid Search RAG

A document question-answering app that retrieves with a dense vector index and a BM25 keyword index, fuses the two lists with reciprocal rank fusion, re-ranks the shortlist with a cross-encoder, and answers from that context only.

## Layout

```
hybrid-rag-project/
├── data/                         # 10 source documents (Markdown, text, and one PDF)
├── evaluation/
│   ├── questions.md              # the 10 questions, written before indexing
│   ├── questions.json            # gold substrings used by the scorer
│   ├── comparison.md             # generated retrieval table
│   └── results.json
├── src/
│   ├── ingestion.py              # load and chunk
│   ├── dense_retriever.py        # Chroma + MiniLM
│   ├── sparse_retriever.py       # BM25
│   ├── hybrid_retriever.py       # reciprocal rank fusion
│   ├── reranker.py               # cross-encoder
│   ├── rag_pipeline.py           # prompt and answer
│   └── evaluation.py
├── app.py                        # Streamlit UI
├── scripts/build_index.py
├── scripts/evaluate.py
├── requirements.txt
└── .env                          # API keys; copy from .env.example
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The imports in this project need the LangChain integration packages as well as the libraries named in the lab brief: `langchain-community`, `langchain-text-splitters`, `langchain-chroma`, and `langchain-huggingface`. They are listed in `requirements.txt`.

Copy `.env.example` to `.env` if you do not already have one. Leave the keys empty to use the grounded extractive answer, which quotes retrieved sentences and cites the file and page. Set `LLM_PROVIDER` to `openai`, `groq`, `gemini`, `anthropic`, or `ollama` and fill in the matching key or base URL to generate the answer with a language model. The prompt is the same either way.

## Build and run

```bash
python scripts/build_index.py
python scripts/evaluate.py
streamlit run app.py
```

The first build downloads two public models:

- `sentence-transformers/all-MiniLM-L6-v2` for dense retrieval
- `cross-encoder/ms-marco-MiniLM-L-6-v2` for re-ranking

`build_index.py` prints how many documents and chunks were created, a sample of the metadata, and a dense query. The Streamlit app rebuilds the index from the sidebar if `data/` changes.

## What the pipeline does

1. Load PDF, text, and Markdown files. Each chunk keeps the file name, page number, document type, and a chunk id.
2. Split on paragraphs, lines, and sentences (800 characters, 150 overlap). Text files are page 1. PDF pages keep their page numbers.
3. Embed chunks into Chroma (cosine space, normalized vectors).
4. Build a BM25 index over the same chunks. Hyphenated identifiers such as `ERR-VEC-404` stay one token.
5. For a question, take the top 10 dense hits and the top 10 BM25 hits.
6. Fuse them with reciprocal rank fusion, constant 60. Rank 0 contributes `1/61`.
7. Re-rank the fused top 10 with the cross-encoder and keep 5.
8. Answer from those 5 chunks only. If the context does not contain the answer, the generator says so.

Metadata filters (document type and file name) are applied to both indexes before fusion, so a rejected chunk never enters the union.

Dense distances from Chroma are converted to cosine similarity (`1 - distance`) so every score in the UI is higher-is-better. Cross-encoder scores are logits and can be negative.

## Evaluation

`evaluation/questions.md` lists the ten questions. `python scripts/evaluate.py` writes `evaluation/comparison.md` and `evaluation/results.json`.

A hit is a retrieved chunk that contains that question's gold substring. Vector, BM25, and hybrid are judged on their top 10. Re-ranking is judged on the 5 chunks it keeps. Recall@5 for the re-ranker means the gold chunk survived that cut.

Run the unit tests without downloading models:

```bash
python -m unittest discover -s tests -v
```

## When hybrid search beats a vector-only index

Measured on the ten questions in `evaluation/comparison.md`:

| Method | Recall@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| Vector-only | 0.80 | 1.00 | 0.883 |
| BM25-only | 0.80 | 1.00 | 0.870 |
| Hybrid (RRF) | 0.70 | 1.00 | 0.850 |
| Hybrid + re-rank | 1.00 | 1.00 | 1.000 |

Seven questions were already rank 1 for every method, because the source uses the same words as the question (`ERR-VEC-404`, `POL-RET-90`, `SLA-15M`, and the three conceptual definitions). Recall@5 is 1.00 everywhere: the gold chunk was always in the top 10, and always survived re-ranking.

Hybrid fusion beat the vector list on one question. For “Which embedding model does the Aurora Search Appliance model ASA-210 ship with?”, vector search ranked the manual’s opening above the checkpoint sentence (gold rank 3). BM25 ranked that sentence 2, because `ASA-210` is rare. Fusion kept rank 2.

Fusion also gave a rank back. On “How can a searcher that understands synonyms still fail to surface a product code?”, the vector list had the explanation at rank 1 and BM25 had it at rank 5, behind the manual’s repeated “product code” ticket instructions. Fusion placed the explanation at rank 2. On the cross-encoder question, BM25 had the model-name chunk at rank 1 and the vector list had it at rank 2; fusion stayed at rank 2.

The cross-encoder is what made the full pipeline better than vector-only. It moved the gold chunk to rank 1 on those three disagreements (q7, q8, q9). Re-ranking cannot recover a chunk that never entered the fused top 10. Here it did not have to: both indexes had already recalled the answer by rank 5, and the joint scorer fixed the order.

