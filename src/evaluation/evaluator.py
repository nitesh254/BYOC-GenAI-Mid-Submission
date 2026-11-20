"""Quantitative evaluation for retrieval accuracy, answer quality, and refusal."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.pipeline.rag_pipeline import RAGPipeline


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "will",
    "with", "you", "your", "our", "we", "can", "if", "not", "this", "they", "their",
    "page", "pdf", "document", "policy",
}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric word tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokenize text excluding common function words."""
    return [tok for tok in tokenize(text) if tok not in STOPWORDS]


def compute_token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 overlap between predicted and expected answers."""
    pred_tokens = content_tokens(prediction)
    ref_tokens = content_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_keyword_recall(prediction: str, expected_keywords: list[str]) -> float:
    """Compute fraction of required factual keywords appearing in the prediction."""
    if not expected_keywords:
        return 1.0
    lower_pred = prediction.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lower_pred)
    return hits / len(expected_keywords)


def compute_context_faithfulness(prediction: str, context_texts: list[str]) -> float:
    """Estimate lexical groundedness: share of answer content tokens present in context."""
    ans_tokens = content_tokens(prediction)
    if not ans_tokens:
        return 0.0
    ctx_vocab = set(content_tokens(" ".join(context_texts)))
    supported = sum(1 for tok in ans_tokens if tok in ctx_vocab)
    return supported / len(ans_tokens)


def evaluate_dataset(
    pipeline: RAGPipeline,
    eval_path: Path,
    top_k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Run retrieval and generation evaluation on a JSON Q&A benchmark file."""
    with eval_path.open("r", encoding="utf-8") as handle:
        items = json.load(handle)

    k_list = sorted(set(top_k_values or [1, 3, 5]))
    max_k = max(k_list)

    answerable_records: list[dict[str, Any]] = []
    unanswerable_records: list[dict[str, Any]] = []
    per_question: list[dict[str, Any]] = []

    for item in items:
        question = item["question"]
        answerable = bool(item.get("answerable", True))
        expected_source = item.get("expected_source", "")
        expected_answer = item.get("expected_answer", "")
        expected_keywords = item.get("expected_keywords", [])

        retrieved_max = pipeline.retrieve(question, top_k=max_k)
        response = pipeline.answer(question)

        if answerable:
            sources_ranked = [r.chunk.source for r in retrieved_max]
            hit_by_k = {
                f"hit@{k}": float(expected_source in sources_ranked[:k])
                for k in k_list
            }
            precision_by_k = {
                f"precision@{k}": (
                    sum(1 for s in sources_ranked[:k] if s == expected_source) / k
                )
                for k in k_list
            }
            mrr = 0.0
            for idx, src in enumerate(sources_ranked, start=1):
                if src == expected_source:
                    mrr = 1.0 / idx
                    break

            kw_acc = compute_keyword_recall(response.answer, expected_keywords)
            tok_f1 = compute_token_f1(response.answer, expected_answer)
            faithfulness = compute_context_faithfulness(
                response.answer,
                [r.chunk.text for r in response.retrieved],
            )
            cited_expected = any(c.source == expected_source for c in response.citations)

            record = {
                "id": item["id"],
                "question": question,
                "answerable": True,
                "expected_source": expected_source,
                "mrr": round(mrr, 4),
                **hit_by_k,
                **precision_by_k,
                "keyword_accuracy": round(kw_acc, 4),
                "token_f1": round(tok_f1, 4),
                "faithfulness_proxy": round(faithfulness, 4),
                "cited_expected_source": cited_expected,
                "abstained": response.abstained,
                "predicted_answer": response.answer,
            }
            answerable_records.append(record)
        else:
            refused_correctly = response.abstained or (
                pipeline.refusal_phrase.lower() in response.answer.lower()
            )
            record = {
                "id": item["id"],
                "question": question,
                "answerable": False,
                "refused_correctly": refused_correctly,
                "abstained": response.abstained,
                "top_similarity": round(
                    retrieved_max[0].score if retrieved_max else 0.0, 4
                ),
                "predicted_answer": response.answer,
            }
            unanswerable_records.append(record)

        per_question.append(record)

    summary: dict[str, Any] = {
        "model": pipeline.llm.model_label,
        "embedding_model": pipeline.embedder.model_name,
        "total_questions": len(items),
        "answerable_count": len(answerable_records),
        "unanswerable_count": len(unanswerable_records),
    }

    if answerable_records:
        n = len(answerable_records)
        for k in k_list:
            summary[f"hit@{k}"] = round(sum(r[f"hit@{k}"] for r in answerable_records) / n, 4)
            summary[f"precision@{k}"] = round(
                sum(r[f"precision@{k}"] for r in answerable_records) / n, 4
            )
        summary["mrr"] = round(sum(r["mrr"] for r in answerable_records) / n, 4)
        summary["keyword_accuracy"] = round(
            sum(r["keyword_accuracy"] for r in answerable_records) / n, 4
        )
        summary["token_f1"] = round(sum(r["token_f1"] for r in answerable_records) / n, 4)
        summary["faithfulness_proxy"] = round(
            sum(r["faithfulness_proxy"] for r in answerable_records) / n, 4
        )
        summary["citation_rate"] = round(
            sum(1 for r in answerable_records if r["cited_expected_source"]) / n, 4
        )

    if unanswerable_records:
        summary["out_of_scope_refusal_rate"] = round(
            sum(1 for r in unanswerable_records if r["refused_correctly"])
            / len(unanswerable_records),
            4,
        )

    return {"summary": summary, "results": per_question}
