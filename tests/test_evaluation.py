"""Unit tests for evaluation metric calculations."""

from src.evaluation.evaluator import (
    compute_context_faithfulness,
    compute_keyword_recall,
    compute_token_f1,
)


def test_compute_token_f1() -> None:
    assert compute_token_f1("30 days server deletion", "30 days server deletion") == 1.0
    assert 0.0 < compute_token_f1("30 days", "30 days and 60 days") < 1.0
    assert compute_token_f1("completely unrelated", "30 days") == 0.0


def test_compute_keyword_recall() -> None:
    assert compute_keyword_recall("Deleted in 30 days and backups in 60 days.", ["30 days", "60 days"]) == 1.0
    assert compute_keyword_recall("Deleted in 30 days.", ["30 days", "60 days"]) == 0.5


def test_compute_context_faithfulness() -> None:
    score_grounded = compute_context_faithfulness(
        "Uptime is 99.99% monthly.",
        ["Basecamp Big guarantees 99.99% monthly uptime."],
    )
    score_hallucinated = compute_context_faithfulness(
        "Maternity leave is 26 weeks paid.",
        ["Basecamp Big guarantees 99.99% monthly uptime."],
    )
    assert score_grounded == 1.0
    assert score_hallucinated == 0.0
