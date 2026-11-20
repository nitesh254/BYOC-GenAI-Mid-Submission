"""CLI entry point to ingest PDFs, ask questions, and run baseline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config, load_prompts, resolve_path
from src.evaluation.evaluator import evaluate_dataset
from src.llm.factory import create_llm_client
from src.pipeline.rag_pipeline import RAGPipeline, ingest_pdfs_to_index
from src.retrieval.embedder import LocalEmbedder
from src.retrieval.vector_store import FaissVectorStore


def build_pipeline(cfg: dict) -> RAGPipeline:
    """Construct a RAGPipeline from loaded YAML configuration."""
    index_dir = resolve_path(cfg["paths"]["index_dir"])
    store = FaissVectorStore.load(index_dir)
    embedder = LocalEmbedder(cfg["retrieval"]["embedding_model"])
    llm = create_llm_client(cfg["llm"])
    prompts = load_prompts()
    return RAGPipeline(
        vector_store=store,
        embedder=embedder,
        llm=llm,
        prompts=prompts,
        top_k=int(cfg["retrieval"]["top_k"]),
        min_similarity_score=float(cfg["retrieval"]["min_similarity_score"]),
        refusal_phrase=str(cfg["evaluation"]["refusal_phrase"]),
    )


def run_ingest(cfg: dict) -> dict:
    """Execute document ingestion and FAISS index creation."""
    embedder = LocalEmbedder(cfg["retrieval"]["embedding_model"])
    stats = ingest_pdfs_to_index(
        pdf_dir=resolve_path(cfg["paths"]["sample_pdf_dir"]),
        index_dir=resolve_path(cfg["paths"]["index_dir"]),
        embedder=embedder,
        chunk_size_words=int(cfg["ingestion"]["chunk_size_words"]),
        chunk_overlap_words=int(cfg["ingestion"]["chunk_overlap_words"]),
        min_chunk_words=int(cfg["ingestion"]["min_chunk_words"]),
    )
    print("Ingestion complete:", json.dumps(stats, indent=2))
    return stats


def run_query(cfg: dict, question: str) -> None:
    """Answer a single user query from the CLI."""
    pipeline = build_pipeline(cfg)
    resp = pipeline.answer(question)
    print(f"\nModel: {resp.model_label}")
    print(f"Question: {resp.question}")
    print(f"Answer:\n{resp.answer}\n")
    if resp.citations:
        print("Citations:")
        for cite in resp.citations:
            print(f"  - {cite.source} (Page {cite.page}, score={cite.score:.3f})")


def run_eval(cfg: dict) -> dict:
    """Run quantitative baseline evaluation and save summary artifact."""
    pipeline = build_pipeline(cfg)
    eval_path = resolve_path(cfg["paths"]["eval_dataset_path"])
    report = evaluate_dataset(
        pipeline=pipeline,
        eval_path=eval_path,
        top_k_values=list(cfg["evaluation"]["top_k_values"]),
    )
    assets_dir = resolve_path(cfg["paths"]["assets_dir"])
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_file = assets_dir / "eval_summary.json"
    with out_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print("Evaluation Summary:\n", json.dumps(report["summary"], indent=2))
    print(f"Saved detailed evaluation report to {out_file}")
    return report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="DocuMind AI - Baseline RAG CLI")
    parser.add_argument(
        "command",
        choices=["ingest", "query", "evaluate", "all"],
        help="Pipeline stage to execute.",
    )
    parser.add_argument("--question", type=str, default="", help="Question for query mode.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config path.")
    return parser.parse_args()


def main() -> None:
    """Dispatch CLI command."""
    args = parse_args()
    cfg = load_config(Path(args.config))

    if args.command == "ingest":
        run_ingest(cfg)
    elif args.command == "query":
        if not args.question.strip():
            raise ValueError("--question is required for query command")
        run_query(cfg, args.question)
    elif args.command == "evaluate":
        run_eval(cfg)
    elif args.command == "all":
        run_ingest(cfg)
        run_query(cfg, "How long does it take to permanently delete content after canceling an account?")
        run_eval(cfg)


if __name__ == "__main__":
    main()
