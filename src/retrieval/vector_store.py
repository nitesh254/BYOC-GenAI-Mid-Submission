"""FAISS vector store with JSON metadata persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.ingestion.chunker import TextChunk


@dataclass(frozen=True)
class RetrievedChunk:
    """Search result pairing a TextChunk with its cosine similarity score."""

    chunk: TextChunk
    score: float
    rank: int


class FaissVectorStore:
    """Stores normalized chunk embeddings in a FAISS inner-product index."""

    INDEX_FILENAME = "index.faiss"
    METADATA_FILENAME = "chunks.json"

    def __init__(self, index: faiss.Index, chunks: list[TextChunk]) -> None:
        self.index = index
        self.chunks = chunks

    @classmethod
    def build(cls, embeddings: np.ndarray, chunks: list[TextChunk]) -> "FaissVectorStore":
        """Build a new FAISS IndexFlatIP from normalized embeddings and chunks."""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks must match number of embedding rows")
        if embeddings.size == 0:
            raise ValueError("Cannot build FAISS index from empty embeddings")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.asarray(embeddings, dtype=np.float32))
        return cls(index=index, chunks=chunks)

    def save(self, directory: Path) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / self.INDEX_FILENAME))
        payload: list[dict[str, Any]] = [c.to_dict() for c in self.chunks]
        with (directory / self.METADATA_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, directory: Path) -> "FaissVectorStore":
        """Load a previously saved FAISS index and chunk metadata."""
        index_path = directory / cls.INDEX_FILENAME
        meta_path = directory / cls.METADATA_FILENAME
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"FAISS index artifacts not found in {directory}")

        index = faiss.read_index(str(index_path))
        with meta_path.open("r", encoding="utf-8") as handle:
            raw_chunks = json.load(handle)
        chunks = [TextChunk(**item) for item in raw_chunks]
        return cls(index=index, chunks=chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[RetrievedChunk]:
        """Return top_k matching chunks ranked by cosine similarity."""
        if top_k <= 0 or not self.chunks:
            return []
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(np.asarray(query_vector, dtype=np.float32), k)

        results: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append(
                RetrievedChunk(chunk=self.chunks[idx], score=float(score), rank=rank)
            )
        return results
