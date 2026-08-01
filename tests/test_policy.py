import pytest

from app.graph.policy import decide
from app.providers.stt import _derive_language
from app.models.schemas import CoverageItem


def _item(competency: str, status: str, evidence: str | None = "quote") -> CoverageItem:
    return CoverageItem(competency=competency, status=status,
                         evidence=None if status == "missing" else evidence)


def test_all_covered_scores_now():
    coverage = [_item("Diagnostic method", "covered"), _item("Caching & performance", "covered")]
    decision = decide(coverage, followups_used=0)
    assert decision.probe is False
    assert decision.target is None
    assert "covered" in decision.reason


def test_single_gap_probes():
    coverage = [
        _item("Diagnostic method", "covered"),
        _item("Async & concurrency", "partial"),
        _item("Caching & performance", "covered"),
    ]
    decision = decide(coverage, followups_used=0)
    assert decision.probe is True
    assert decision.target == "Async & concurrency"
    assert decision.reason


def test_two_missing_skips():
    coverage = [
        _item("Diagnostic method", "missing"),
        _item("Async & concurrency", "missing"),
        _item("Caching & performance", "covered"),
    ]
    decision = decide(coverage, followups_used=0)
    assert decision.probe is False
    assert decision.target is None
    assert decision.reason


def test_budget_spent_skips_even_with_single_gap():
    coverage = [
        _item("Diagnostic method", "covered"),
        _item("Async & concurrency", "partial"),
    ]
    decision = decide(coverage, followups_used=1)
    assert decision.probe is False
    assert decision.target is None
    assert "budget" in decision.reason


@pytest.mark.parametrize("text, expected", [
    ("هذا كلام عربي بدون أي كلمات إنجليزية", "ar"),
    ("This is a plain English answer about thread pools.", "en"),
    ("هو موضوع الـ thread pool و الـ N+1 queries معقد شوية", "ar-en"),
])
def test_derive_language(text, expected):
    assert _derive_language(text) == expected
