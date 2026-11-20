"""Overlapping word-window chunking with source and page metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.ingestion.pdf_loader import PageDocument


@dataclass(frozen=True)
class TextChunk:
    """A single indexed text chunk with provenance metadata."""

    chunk_id: str
    source: str
    page: int
    chunk_index: int
    text: str
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk to a JSON-compatible dictionary."""
        return asdict(self)


def chunk_page(
    page_doc: PageDocument,
    chunk_size_words: int = 180,
    chunk_overlap_words: int = 35,
    min_chunk_words: int = 20,
) -> list[TextChunk]:
    """Split a PageDocument into overlapping word-window TextChunks."""
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if chunk_overlap_words < 0 or chunk_overlap_words >= chunk_size_words:
        raise ValueError("chunk_overlap_words must be >= 0 and < chunk_size_words")

    words = page_doc.text.split()
    if not words:
        return []

    step = chunk_size_words - chunk_overlap_words
    chunks: list[TextChunk] = []
    start = 0
    chunk_idx = 0

    while start < len(words):
        window = words[start : start + chunk_size_words]
        if len(window) < min_chunk_words and chunks:
            break
        chunk_text = " ".join(window)
        chunk_id = f"{page_doc.source}::p{page_doc.page}::c{chunk_idx}"
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                source=page_doc.source,
                page=page_doc.page,
                chunk_index=chunk_idx,
                text=chunk_text,
                word_count=len(window),
            )
        )
        if start + chunk_size_words >= len(words):
            break
        start += step
        chunk_idx += 1

    return chunks


def chunk_documents(
    pages: list[PageDocument],
    chunk_size_words: int = 180,
    chunk_overlap_words: int = 35,
    min_chunk_words: int = 20,
) -> list[TextChunk]:
    """Chunk a list of PageDocument objects across multiple PDFs."""
    all_chunks: list[TextChunk] = []
    for page_doc in pages:
        all_chunks.extend(
            chunk_page(
                page_doc,
                chunk_size_words=chunk_size_words,
                chunk_overlap_words=chunk_overlap_words,
                min_chunk_words=min_chunk_words,
            )
        )
    return all_chunks
