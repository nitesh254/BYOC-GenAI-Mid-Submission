"""Unit tests for FAISS vector store build, save/load, and search."""

from pathlib import Path

import numpy as np

from src.ingestion.chunker import TextChunk
from src.retrieval.vector_store import FaissVectorStore


def test_faiss_build_save_load_and_search(tmp_path: Path) -> None:
    chunks = [
        TextChunk("a.pdf::p1::c0", "a.pdf", 1, 0, "refund policy full refund", 4),
        TextChunk("b.pdf::p2::c0", "b.pdf", 2, 0, "uptime sla guarantee 99.99", 4),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    store = FaissVectorStore.build(embeddings, chunks)
    store.save(tmp_path)

    loaded = FaissVectorStore.load(tmp_path)
    query = np.array([[0.05, 0.99, 0.0, 0.0]], dtype=np.float32)
    results = loaded.search(query, top_k=2)

    assert len(results) == 2
    assert results[0].chunk.source == "b.pdf"
    assert results[0].chunk.page == 2
    assert results[0].rank == 1
