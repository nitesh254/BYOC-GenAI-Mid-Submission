# Planned Experiments for Final Submission

While Version 1 establishes a reproducible single-configuration baseline, the final submission will systematically evaluate four experimental axes on a 50-question manually verified benchmark:

## Experiment A — Chunk Size Sensitivity
- **Configurations:** `256`, `512`, and `1024` tokens (with ~15% overlap).
- **Hypothesis:** Smaller chunks (`256`) improve `Precision@K` for specific numeric facts (e.g., `30 days`, `99.99%`), whereas larger chunks (`512`–`1024`) preserve multi-clause policy context.

## Experiment B — Embedding Model Comparison
- **Models:** `sentence-transformers/all-MiniLM-L6-v2` (384-d, ~90 MB) vs. `BAAI/bge-small-en-v1.5` (384-d, ~130 MB).
- **Metrics:** `Hit@1`, `Hit@3`, `MRR`, and query latency on CPU.

## Experiment C — Retrieval Depth (`Top-K`)
- **Settings:** `K = 3`, `K = 5`, `K = 10`.
- **Goal:** Measure the trade-off between higher `Recall@K` and context distraction in compact local LLMs (0.5B–3B parameters).

## Experiment D — Ablation Study (No-RAG vs. Baseline RAG vs. Tuned RAG)
1. **Basic Local LLM (No Retrieval):** Quantifies baseline hallucination rate and factual errors on company-specific policies.
2. **Baseline RAG (V1):** Dense FAISS retrieval with default chunking and prompt rules.
3. **Improved RAG (V2):** Hybrid BM25 + FAISS retrieval + experimentally optimal chunk size, embedding model, and `Top-K`.
