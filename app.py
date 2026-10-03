"""Streamlit interface for the hybrid search RAG pipeline."""

from __future__ import annotations

from src.bootstrap import configure_environment

configure_environment()

import streamlit as st
from dotenv import load_dotenv

from src.evaluation import load_questions
from src.paths import QUESTIONS_PATH, ROOT
from src.rag_pipeline import RAGPipeline
from src.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DENSE_K,
    EMBEDDING_MODEL,
    FUSED_K,
    RERANK_K,
    RERANKER_MODEL,
    RRF_CONSTANT,
    SPARSE_K,
)

load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Hybrid Search RAG", layout="wide")
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.4rem; max-width: 1180px; }
        div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the hybrid index, embeddings, and cross-encoder...")
def load_pipeline() -> RAGPipeline:
    return RAGPipeline.load()


def main() -> None:
    pipeline = load_pipeline()
    file_names = sorted({chunk.metadata.get("file_name", "") for chunk in pipeline.chunks})
    document_types = sorted({chunk.metadata.get("document_type", "") for chunk in pipeline.chunks})
    examples = _example_questions()

    st.title("Hybrid Search RAG")
    st.caption(
        "Dense vectors, BM25, reciprocal rank fusion, and a cross-encoder. "
        "Answers stay inside the retrieved chunks."
    )

    with st.sidebar:
        st.header("Index")
        st.write(f"{len(pipeline.chunks)} chunks from {len(file_names)} files")
        st.caption(f"Embeddings: `{EMBEDDING_MODEL}`")
        st.caption(f"Re-ranker: `{RERANKER_MODEL}`")
        st.caption(f"Chunks: {CHUNK_SIZE} characters, overlap {CHUNK_OVERLAP}. RRF constant {RRF_CONSTANT}.")
        st.caption(f"Generator: `{pipeline.generator.backend}`")
        if st.button("Rebuild index", use_container_width=True):
            with st.spinner("Rebuilding the vector store and BM25 index..."):
                RAGPipeline.build()
            load_pipeline.clear()
            st.rerun()

        st.header("Metadata filter")
        document_type = st.selectbox("Document type", ["all", *document_types])
        file_name = st.selectbox("File", ["all", *file_names])

        with st.expander("Retrieval depth"):
            dense_k = st.slider("Dense results", 3, 20, DENSE_K)
            sparse_k = st.slider("BM25 results", 3, 20, SPARSE_K)
            fused_k = st.slider("Fused results", 3, 20, FUSED_K)
            rerank_k = st.slider("Re-ranked results", 1, 10, RERANK_K)

    if "question" not in st.session_state:
        st.session_state.question = examples[0] if examples else ""

    def apply_preset() -> None:
        if st.session_state.preset != "Custom":
            st.session_state.question = st.session_state.preset

    st.selectbox(
        "Evaluation question",
        ["Custom", *examples],
        key="preset",
        on_change=apply_preset,
    )
    st.text_area("Question", key="question", height=100)
    search = st.button("Search", type="primary")

    if not search:
        st.info("Ask a question to compare dense, sparse, fused, and re-ranked chunks.")
        return

    question = st.session_state.question.strip()
    if not question:
        st.warning("Enter a question first.")
        return

    metadata_filter = {}
    if document_type != "all":
        metadata_filter["document_type"] = document_type
    if file_name != "all":
        metadata_filter["file_name"] = file_name

    with st.spinner("Retrieving, fusing, re-ranking, and writing the answer..."):
        try:
            result = pipeline.answer(
                question,
                metadata_filter=metadata_filter or None,
                dense_k=dense_k,
                sparse_k=sparse_k,
                fused_k=fused_k,
                rerank_k=rerank_k,
            )
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            return

    if result["generator_warning"]:
        st.warning(result["generator_warning"])

    timings = result["timings_ms"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Response latency", f"{result['latency_ms']:.0f} ms")
    metric_cols[1].metric("Dense", f"{timings['dense_ms']:.0f} ms")
    metric_cols[2].metric("BM25", f"{timings['sparse_ms']:.0f} ms")
    metric_cols[3].metric("Re-rank + answer", f"{timings['rerank_ms'] + timings['generate_ms']:.0f} ms")
    st.caption(
        "Generator: `{generator}`. Fusion took {fusion:.1f} ms. "
        "Dense scores are cosine similarity. BM25 and fusion scores are higher-is-better. "
        "Re-rank scores are cross-encoder logits.".format(
            generator=result["generator"],
            fusion=timings["fusion_ms"],
        )
    )

    st.subheader("Answer")
    with st.container(border=True):
        st.markdown(_escape_markdown(result["answer"]))

    sources, dense, sparse, fused, reranked = st.tabs(
        ["Sources", "Dense", "BM25", "Fusion", "Re-rank"]
    )
    with sources:
        if not result["reranked_results"]:
            st.write("No chunks were retrieved.")
        for index, hit in enumerate(result["reranked_results"], start=1):
            document = hit["document"]
            title = (
                f"Source {index}: {document.metadata.get('file_name')} "
                f"· Page {document.metadata.get('page_number')} "
                f"· score {hit['score']:.4f}"
            )
            with st.expander(title, expanded=index == 1):
                st.caption(document.metadata.get("chunk_id", ""))
                st.markdown(_escape_markdown(document.page_content))
    with dense:
        st.dataframe(_rows(result["dense_results"]), hide_index=True, use_container_width=True)
    with sparse:
        st.dataframe(_rows(result["sparse_results"]), hide_index=True, use_container_width=True)
    with fused:
        st.dataframe(_rows(result["fused_results"]), hide_index=True, use_container_width=True)
    with reranked:
        st.dataframe(_rows(result["reranked_results"]), hide_index=True, use_container_width=True)

    with st.expander("Prompt sent to the generator"):
        st.code(result["prompt"], language="text")


def _escape_markdown(text: str) -> str:
    """Show retrieved text as prose. Source files contain Markdown headings."""

    return text.replace("\\", "\\\\").replace("#", "\\#")


def _rows(results: list[dict]) -> list[dict]:
    rows = []
    for rank, result in enumerate(results, start=1):
        document = result["document"]
        rows.append(
            {
                "rank": rank,
                "score": round(float(result["score"]), 4),
                "file": document.metadata.get("file_name"),
                "page": document.metadata.get("page_number"),
                "chunk_id": document.metadata.get("chunk_id"),
                "snippet": document.page_content[:220].replace("\n", " "),
            }
        )
    return rows


def _example_questions() -> list[str]:
    if not QUESTIONS_PATH.exists():
        return []
    return [item["question"] for item in load_questions()]


if __name__ == "__main__":
    main()
else:
    main()
