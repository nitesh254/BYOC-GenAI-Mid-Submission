# DocuMind AI — Modular RAG Policy & Document Assistant (Version 1 Baseline)

## Project Overview
**DocuMind AI** is a modular, end-to-end Retrieval-Augmented Generation (RAG) document question-answering system designed to run completely locally using open-source models. Given a collection of PDF policy documents, it extracts page-level text, creates overlapping chunks, indexes dense vector embeddings in FAISS, retrieves relevant passages, and generates grounded answers with explicit document name and page citations—while abstaining on out-of-scope questions.

## Business Objective
Customer support, compliance, and operations teams spend significant time searching across fragmented policy documents (refund rules, cancellation terms, SLAs, privacy disclosures, and tax policies). Manual lookup is slow and inconsistent, whereas ungrounded LLMs hallucinate company policies.  
DocuMind AI solves this by:
- Grounding every response in retrieved policy excerpts with **document + page citations**.
- Running **100% locally** with open-source embeddings and LLMs (zero external API keys or data leakage).
- Establishing a reproducible quantitative baseline (`Hit@K`, `Precision@K`, `MRR`, `Keyword Accuracy`, `Token F1`, `Faithfulness Proxy`, and `Out-of-Scope Refusal Rate`) for systematic experimentation.

## Dataset Source and Licence
- **Source:** [37signals / Basecamp Open Policies (`basecamp/policies`)](https://github.com/basecamp/policies)
- **Licence:** **Creative Commons Attribution 4.0 International (CC BY 4.0)** — full licence included in [`data/raw/basecamp_policies/LICENSE.md`](data/raw/basecamp_policies/LICENSE.md).
- **Bundled Sample PDFs (`data/sample/`):** Generated reproducibly from the raw markdown policies via `scripts/build_sample_pdfs.py`:
  1. `01_refund_and_cancellation_policy.pdf`
  2. `02_security_and_sla_policy.pdf`
  3. `03_privacy_policy.pdf`
  4. `04_taxes_and_acceptable_use_policy.pdf`
- **Evaluation Benchmark (`data/eval/eval_questions.json`):** 20 manually verified questions (16 answerable across the 4 policy PDFs + 4 unanswerable out-of-scope questions for hallucination/refusal evaluation).

## Repository Structure
```text
documind-ai/
├── app.py                              # Streamlit web interface calling src/ modules
├── configs/
│   ├── config.yaml                     # Paths, chunking, embedding, FAISS, and LLM config
│   └── prompts.yaml                    # Versioned RAG system & user prompt templates
├── data/
│   ├── raw/basecamp_policies/          # Raw CC BY 4.0 markdown sources + LICENSE.md
│   ├── sample/                         # Bundled multi-page sample PDFs
│   ├── processed/                      # Persisted FAISS index & chunk metadata
│   └── eval/eval_questions.json        # 20 verified Q&A benchmark items
├── docs/
│   ├── architecture.md                 # System diagram & V2 roadmap
│   └── experiments_plan.md             # Planned Experiments A–D for final submission
├── notebooks/
│   └── 01_baseline_exploration.ipynb   # Corpus & chunk statistics inspection
├── scripts/
│   └── build_sample_pdfs.py            # Converts raw CC BY 4.0 markdown into sample PDFs
├── src/
│   ├── config.py                       # YAML configuration & prompt loader
│   ├── ingestion/                      # PDF text extraction & overlapping chunker
│   ├── retrieval/                      # Local SentenceTransformer embedder & FAISS store
│   ├── llm/                            # Local Ollama & Hugging Face LLM adapters
│   ├── pipeline/                       # End-to-end RAG ingestion, retrieval & citation logic
│   ├── evaluation/                     # Retrieval, answer quality & refusal metrics
│   └── main.py                         # CLI entry point (ingest / query / evaluate / all)
├── tests/                              # Automated pytest suite
├── assets/                             # Generated evaluation reports (eval_summary.json)
└── requirements.txt
```

## Installation Instructions
Requires **Python 3.10+** (tested on Python 3.12, Windows). No API keys are needed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

*(Optional)* If you want to run via **Ollama** instead of the built-in Hugging Face local backend:
```powershell
ollama pull qwen2.5:1.5b
```
By default (`llm.provider: "auto"` in `configs/config.yaml`), DocuMind AI automatically uses Ollama if `qwen2.5:1.5b` is running locally, and otherwise falls back seamlessly to local `Qwen/Qwen2.5-0.5B-Instruct` via Hugging Face `transformers`.

## Execution Steps
Run the complete baseline pipeline (ingest sample PDFs → run sample query → evaluate on 20 questions):
```powershell
python -m src.main all
```

Or run individual stages:
```powershell
python -m src.main ingest
python -m src.main query --question "What monthly uptime does the Basecamp Big SLA guarantee?"
python -m src.main evaluate
```

Run automated unit tests:
```powershell
pytest -v
```

Launch the interactive Streamlit UI:
```powershell
streamlit run app.py
```

## Current Implementation Status (Version 1 Baseline)
- Modular PDF extraction with 1-based page tracking (`src/ingestion/pdf_loader.py`).
- Overlapping word-window chunking with source/page provenance (`src/ingestion/chunker.py`).
- Local `all-MiniLM-L6-v2` embeddings + `FAISS IndexFlatIP` persistence (`src/retrieval/`).
- Config-driven local open-source LLM backend (`Ollama` + `Hugging Face` fallback).
- Similarity-threshold abstention guard (`0.35`) + source/page citation formatting (`src/pipeline/rag_pipeline.py`).
- Quantitative evaluation harness + unit test suite (`src/evaluation/`, `tests/` — 10/10 tests passing).

### Baseline Evaluation Results (`assets/eval_summary.json`)
Evaluated on 20 verified benchmark questions (16 answerable, 4 out-of-scope) using `sentence-transformers/all-MiniLM-L6-v2` + `Qwen/Qwen2.5-0.5B-Instruct`:

| Metric | V1 Baseline Score |
| :--- | :--- |
| `Hit@1` / `Hit@3` / `Hit@5` | `0.875` / `1.000` / `1.000` |
| `Precision@1` / `Precision@3` / `Precision@5` | `0.875` / `0.8125` / `0.700` |
| `MRR` | `0.9375` |
| `Keyword Accuracy` | `0.8438` |
| `Token F1` | `0.6112` |
| `Faithfulness Proxy` | `0.8981` |
| `Citation Rate` | `1.000` |
| `Out-of-Scope Refusal Rate` | `1.000` |

## Known Limitations
1. **Single-turn stateless queries:** Follow-up questions do not yet carry conversational history.
2. **Dense-only retrieval:** Pure semantic similarity can occasionally miss exact acronyms or legal article numbers (e.g., `ITFA`, `Article 49(1)(b)`).
3. **Lightweight V1 local LLM:** Compact 0.5B–1.5B models fit comfortably within 8 GB RAM / 3 GB VRAM, but can be sensitive to noisy or overly long retrieved contexts.
4. **Lexical faithfulness proxy:** V1 faithfulness checks lexical overlap with retrieved context rather than full NLI entailment.

## Planned Improvements for Final Submission
See [`docs/architecture.md`](docs/architecture.md) and [`docs/experiments_plan.md`](docs/experiments_plan.md):
- Scale ground-truth evaluation from 20 to **50 manually verified Q&A pairs**.
- Execute **Experiments A–D** (chunk sizes `256/512/1024`, embedding comparison `all-MiniLM-L6-v2` vs `bge-small-en-v1.5`, `Top-3/5/10`, and **Basic LLM vs. Baseline RAG vs. Improved RAG**).
- Add **Hybrid BM25 + FAISS retrieval** and lightweight cross-encoder reranking.
- Add **multi-turn conversation memory** and an **evaluation comparison tab** inside Streamlit.
