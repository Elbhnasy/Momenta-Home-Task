import pytest
from pydantic import ValidationError

from app.graph.judge import _Verdict
from app.models.schemas import CoverageItem, Evaluation


@pytest.mark.parametrize("status,evidence", [
    ("missing", "claimed evidence"),
    ("covered", None),
    ("partial", "   "),
])
def test_coverage_evidence_invariant(status, evidence):
    with pytest.raises(ValidationError):
        CoverageItem(competency="Diagnostic method", status=status, evidence=evidence)


@pytest.mark.parametrize("score", [0, 6, True, "3"])
def test_provider_verdict_score_is_strictly_one_to_five(score):
    with pytest.raises(ValidationError):
        _Verdict(score=score, justification="one line")


@pytest.mark.parametrize("score", [0, 6, True, "3"])
def test_api_evaluation_score_is_strictly_one_to_five(score):
    with pytest.raises(ValidationError):
        Evaluation(
            claims_summary_en="summary",
            retrieved=[],
            coverage=[],
            decision={"probe": False, "target": None, "reason": "done"},
            score=score,
            justification="one line",
            followup_question=None,
        )
