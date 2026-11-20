"""PDF text extraction preserving document name and 1-based page numbers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PageDocument:
    """Represents cleaned text extracted from a single PDF page."""

    source: str
    page: int
    text: str


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip trailing page footer artifacts."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(pdf_path: Path) -> list[PageDocument]:
    """Extract cleaned text per page from a single PDF file."""
    reader = PdfReader(str(pdf_path))
    pages: list[PageDocument] = []
    for idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = normalize_whitespace(raw_text)
        if cleaned:
            pages.append(PageDocument(source=pdf_path.name, page=idx, text=cleaned))
    return pages


def load_pdfs_from_directory(directory: Path) -> list[PageDocument]:
    """Load all PDF files sorted by filename from a directory."""
    if not directory.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {directory}")

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {directory}")

    all_pages: list[PageDocument] = []
    for pdf_file in pdf_files:
        all_pages.extend(load_pdf(pdf_file))
    return all_pages
