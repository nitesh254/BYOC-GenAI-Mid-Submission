"""Unit tests for PDF loading and overlapping text chunking."""

from pathlib import Path

import pytest

from src.ingestion.chunker import chunk_documents, chunk_page
from src.ingestion.pdf_loader import PageDocument, load_pdfs_from_directory, normalize_whitespace


def test_normalize_whitespace() -> None:
    raw = "Hello   world\t!\n\n\n\nNext   paragraph\x00"
    assert normalize_whitespace(raw) == "Hello world !\n\nNext paragraph"


def test_chunk_page_overlap_and_metadata() -> None:
    words = [f"word{i}" for i in range(50)]
    page = PageDocument(source="sample.pdf", page=2, text=" ".join(words))
    chunks = chunk_page(page, chunk_size_words=20, chunk_overlap_words=5, min_chunk_words=5)

    assert len(chunks) == 3
    assert chunks[0].chunk_id == "sample.pdf::p2::c0"
    assert chunks[0].source == "sample.pdf"
    assert chunks[0].page == 2
    assert chunks[1].text.startswith("word15 word16")


def test_chunk_page_rejects_invalid_overlap() -> None:
    page = PageDocument(source="a.pdf", page=1, text="hello world")
    with pytest.raises(ValueError):
        chunk_page(page, chunk_size_words=10, chunk_overlap_words=10)


def test_load_sample_pdfs_exist() -> None:
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    if not sample_dir.exists():
        pytest.skip("Sample PDFs not built yet")
    pages = load_pdfs_from_directory(sample_dir)
    assert len(pages) >= 4
    chunks = chunk_documents(pages, chunk_size_words=180, chunk_overlap_words=35)
    assert len(chunks) >= 4
