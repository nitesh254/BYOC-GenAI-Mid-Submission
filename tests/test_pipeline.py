"""Unit tests for RAGPipeline prompt formatting, citations, and refusal guarding."""

import numpy as np

from src.config import PromptTemplates
from src.ingestion.chunker import TextChunk
from src.llm.base import BaseLLM
from src.pipeline.rag_pipeline import DEFAULT_REFUSAL, RAGPipeline
from src.retrieval.vector_store import FaissVectorStore


class DummyEmbedder:
    """Deterministic stub embedder returning unit vectors."""

    model_name = "dummy-embedder"

    def embed_query(self, query: str) -> np.ndarray:
        if "salary" in query.lower():
            return np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)


class DummyLLM(BaseLLM):
    """Deterministic stub LLM verifying prompt input."""

    model_label = "dummy/local-test"

    def __init__(self) -> None:
        self.called = False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.called = True
        assert "Document Excerpts:" in user_prompt
        return "Uptime is 99.99% [sla.pdf, Page 1]."


def _make_pipeline(min_score: float = 0.2) -> tuple[RAGPipeline, DummyLLM]:
    chunks = [
        TextChunk("sla.pdf::p1::c0", "sla.pdf", 1, 0, "Basecamp Big guarantees 99.99% uptime.", 5)
    ]
    vecs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    store = FaissVectorStore.build(vecs, chunks)
    llm = DummyLLM()
    prompts = PromptTemplates(
        rag_system_prompt="Answer only from excerpts.",
        rag_user_prompt="Document Excerpts:\n{context}\n\nQuestion: {question}",
    )
    pipeline = RAGPipeline(
        vector_store=store,
        embedder=DummyEmbedder(),  # type: ignore[arg-type]
        llm=llm,
        prompts=prompts,
        top_k=1,
        min_similarity_score=min_score,
    )
    return pipeline, llm


def test_pipeline_returns_answer_with_citation() -> None:
    pipeline, llm = _make_pipeline(min_score=0.2)
    resp = pipeline.answer("What is the uptime guarantee?")
    assert llm.called is True
    assert not resp.abstained
    assert "99.99%" in resp.answer
    assert len(resp.citations) == 1
    assert resp.citations[0].source == "sla.pdf"
    assert resp.citations[0].page == 1


def test_pipeline_abstains_below_similarity_threshold() -> None:
    pipeline, llm = _make_pipeline(min_score=0.5)
    resp = pipeline.answer("What is the salary band?")
    assert llm.called is False
    assert resp.abstained is True
    assert resp.answer == DEFAULT_REFUSAL
    assert resp.citations == []
