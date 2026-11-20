"""Streamlit web UI for DocuMind AI (Version 1 Baseline)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from src.config import load_config, load_prompts, resolve_path
from src.llm.factory import create_llm_client
from src.pipeline.rag_pipeline import RAGPipeline, ingest_pdfs_to_index
from src.retrieval.embedder import LocalEmbedder
from src.retrieval.vector_store import FaissVectorStore


st.set_page_config(page_title="DocuMind AI — Policy RAG Assistant", layout="wide")


@st.cache_resource
def get_embedder(model_name: str) -> LocalEmbedder:
    """Cache the local SentenceTransformer embedder across reruns."""
    return LocalEmbedder(model_name)


@st.cache_resource
def get_llm(llm_cfg_json: str):
    """Cache the selected local LLM client across reruns."""
    return create_llm_client(json.loads(llm_cfg_json))


def main() -> None:
    """Render the DocuMind AI Streamlit application."""
    cfg = load_config()
    prompts = load_prompts()
    index_dir = resolve_path(cfg["paths"]["index_dir"])
    sample_dir = resolve_path(cfg["paths"]["sample_pdf_dir"])

    st.title("DocuMind AI — Local RAG Policy Assistant (V1 Baseline)")
    st.caption(
        "Ask questions across PDF policy documents using local embeddings (FAISS) "
        "and a local open-source LLM with source & page citations."
    )

    with st.sidebar:
        st.header("Knowledge Base")
        top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=int(cfg["retrieval"]["top_k"]))
        uploaded_files = st.file_uploader(
            "Upload custom PDFs (optional)",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Rebuild Index from Sample PDFs", use_container_width=True):
            embedder = get_embedder(cfg["retrieval"]["embedding_model"])
            stats = ingest_pdfs_to_index(
                pdf_dir=sample_dir,
                index_dir=index_dir,
                embedder=embedder,
                chunk_size_words=int(cfg["ingestion"]["chunk_size_words"]),
                chunk_overlap_words=int(cfg["ingestion"]["chunk_overlap_words"]),
                min_chunk_words=int(cfg["ingestion"]["min_chunk_words"]),
            )
            st.success(f"Indexed {stats['num_chunks']} chunks from {stats['num_pdfs']} PDFs.")

        if uploaded_files and st.button("Index Uploaded PDFs", use_container_width=True):
            embedder = get_embedder(cfg["retrieval"]["embedding_model"])
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                for uf in uploaded_files:
                    (tmp_path / uf.name).write_bytes(uf.getvalue())
                stats = ingest_pdfs_to_index(
                    pdf_dir=tmp_path,
                    index_dir=index_dir,
                    embedder=embedder,
                    chunk_size_words=int(cfg["ingestion"]["chunk_size_words"]),
                    chunk_overlap_words=int(cfg["ingestion"]["chunk_overlap_words"]),
                    min_chunk_words=int(cfg["ingestion"]["min_chunk_words"]),
                )
            st.success(f"Indexed {stats['num_chunks']} chunks from uploaded PDFs.")

        eval_summary_path = resolve_path(cfg["paths"]["assets_dir"]) / "eval_summary.json"
        if eval_summary_path.exists():
            st.divider()
            st.subheader("Latest Baseline Metrics")
            summary = json.loads(eval_summary_path.read_text(encoding="utf-8")).get("summary", {})
            st.json(summary)

    if not (index_dir / FaissVectorStore.INDEX_FILENAME).exists():
        st.info("Building initial FAISS index from bundled sample PDFs...")
        embedder = get_embedder(cfg["retrieval"]["embedding_model"])
        ingest_pdfs_to_index(
            pdf_dir=sample_dir,
            index_dir=index_dir,
            embedder=embedder,
            chunk_size_words=int(cfg["ingestion"]["chunk_size_words"]),
            chunk_overlap_words=int(cfg["ingestion"]["chunk_overlap_words"]),
            min_chunk_words=int(cfg["ingestion"]["min_chunk_words"]),
        )

    question = st.text_input(
        "Ask a question about the policy documents:",
        placeholder="e.g., What is the monthly uptime guarantee in the Basecamp Big SLA?",
    )

    if question:
        with st.spinner("Retrieving relevant passages and generating grounded answer..."):
            store = FaissVectorStore.load(index_dir)
            embedder = get_embedder(cfg["retrieval"]["embedding_model"])
            llm = get_llm(json.dumps(cfg["llm"], sort_keys=True))
            pipeline = RAGPipeline(
                vector_store=store,
                embedder=embedder,
                llm=llm,
                prompts=prompts,
                top_k=top_k,
                min_similarity_score=float(cfg["retrieval"]["min_similarity_score"]),
                refusal_phrase=str(cfg["evaluation"]["refusal_phrase"]),
            )
            response = pipeline.answer(question, top_k=top_k)

        st.subheader("Answer")
        st.write(response.answer)
        st.caption(f"Backend: `{response.model_label}` | Abstained: `{response.abstained}`")

        if response.citations:
            st.subheader("Sources & Page Citations")
            for cite in response.citations:
                with st.expander(f"{cite.source} — Page {cite.page} (similarity: {cite.score:.3f})"):
                    st.write(cite.excerpt)


if __name__ == "__main__":
    main()
