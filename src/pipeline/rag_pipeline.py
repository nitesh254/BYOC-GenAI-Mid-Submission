"""End-to-end ingestion, retrieval, prompt formatting, and grounded generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import PromptTemplates
from src.ingestion.chunker import chunk_documents
from src.ingestion.pdf_loader import load_pdfs_from_directory
from src.llm.base import BaseLLM
from src.retrieval.embedder import LocalEmbedder
from src.retrieval.vector_store import FaissVectorStore, RetrievedChunk


DEFAULT_REFUSAL = "I could not find this information in the provided documents."


@dataclass(frozen=True)
class Citation:
    """Source document and page citation for a retrieved excerpt."""

    source: str
    page: int
    score: float
    excerpt: str


@dataclass(frozen=True)
class RAGResponse:
    """Structured output returned by the RAG pipeline."""

    question: str
    answer: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    model_label: str
    abstained: bool


def ingest_pdfs_to_index(
    pdf_dir: Path,
    index_dir: Path,
    embedder: LocalEmbedder,
    chunk_size_words: int = 180,
    chunk_overlap_words: int = 35,
    min_chunk_words: int = 20,
) -> dict[str, Any]:
    """Load PDFs, chunk pages, embed chunks, and persist the FAISS vector store."""
    pages = load_pdfs_from_directory(pdf_dir)
    chunks = chunk_documents(
        pages,
        chunk_size_words=chunk_size_words,
        chunk_overlap_words=chunk_overlap_words,
        min_chunk_words=min_chunk_words,
    )
    embeddings = embedder.embed_texts([c.text for c in chunks])
    store = FaissVectorStore.build(embeddings, chunks)
    store.save(index_dir)

    distinct_sources = sorted({p.source for p in pages})
    return {
        "num_pdfs": len(distinct_sources),
        "pdf_files": distinct_sources,
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "index_dir": str(index_dir),
    }


def format_context(retrieved: list[RetrievedChunk]) -> str:
    """Format retrieved chunks with explicit filename and page citations."""
    blocks = []
    for idx, item in enumerate(retrieved, start=1):
        header = f"Excerpt {idx} — File: {item.chunk.source}, Page {item.chunk.page}"
        blocks.append(f"{header}\n{item.chunk.text}")
    return "\n\n---\n\n".join(blocks)


def deduplicate_citations(retrieved: list[RetrievedChunk]) -> list[Citation]:
    """Build unique (source, page) citations preserving retrieval rank order."""
    seen: set[tuple[str, int]] = set()
    citations: list[Citation] = []
    for item in retrieved:
        key = (item.chunk.source, item.chunk.page)
        if key in seen:
            continue
        seen.add(key)
        snippet = item.chunk.text[:220].rstrip() + ("..." if len(item.chunk.text) > 220 else "")
        citations.append(
            Citation(
                source=item.chunk.source,
                page=item.chunk.page,
                score=item.score,
                excerpt=snippet,
            )
        )
    return citations


class RAGPipeline:
    """Coordinates vector retrieval and grounded LLM answer generation."""

    def __init__(
        self,
        vector_store: FaissVectorStore,
        embedder: LocalEmbedder,
        llm: BaseLLM,
        prompts: PromptTemplates,
        top_k: int = 4,
        min_similarity_score: float = 0.20,
        refusal_phrase: str = DEFAULT_REFUSAL,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm
        self.prompts = prompts
        self.top_k = top_k
        self.min_similarity_score = min_similarity_score
        self.refusal_phrase = refusal_phrase

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve the top-k most similar document chunks for a question."""
        cleaned = question.strip()
        if not cleaned:
            return []
        query_vec = self.embedder.embed_query(cleaned)
        return self.vector_store.search(query_vec, top_k=top_k or self.top_k)

    def answer(self, question: str, top_k: int | None = None) -> RAGResponse:
        """Answer a user question with citations and out-of-scope refusal guarding."""
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved or retrieved[0].score < self.min_similarity_score:
            return RAGResponse(
                question=question,
                answer=self.refusal_phrase,
                citations=[],
                retrieved=retrieved,
                model_label=self.llm.model_label,
                abstained=True,
            )

        context_str = format_context(retrieved)
        user_prompt = self.prompts.rag_user_prompt.format(
            context=context_str,
            question=question.strip(),
        )
        generated = self.llm.generate(self.prompts.rag_system_prompt, user_prompt)
        abstained = self.refusal_phrase.lower() in generated.lower()
        citations = [] if abstained else deduplicate_citations(retrieved)

        return RAGResponse(
            question=question,
            answer=generated,
            citations=citations,
            retrieved=retrieved,
            model_label=self.llm.model_label,
            abstained=abstained,
        )
