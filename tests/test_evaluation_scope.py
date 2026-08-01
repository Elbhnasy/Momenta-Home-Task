import json
from pathlib import Path

from app.graph import judge
from app.models.schemas import Chunk, CoverageItem
from app.rag import retrieval


def test_all_rubric_competencies_are_mandatory_even_when_not_retrieved(monkeypatch):
    captured = {}

    def fake_complete(system, user, model_cls, **kwargs):
        captured["user"] = user
        return model_cls(items=[
            CoverageItem(
                competency="Diagnostic method",
                status="covered",
                evidence="I measured it",
            )
        ])

    monkeypatch.setattr(judge.llm, "complete", fake_complete)
    supplemental = [Chunk(
        doc_id="02_dependency_injection.md",
        competency="Dependency injection",
        text="irrelevant supplemental reference",
        score=0.9,
    )]

    coverage = judge.analyze_coverage(
        ["I measured it"], supplemental, "en", thread_id="scope-test"
    )

    expected = [chunk.competency for chunk in retrieval.rubric_chunks()]
    assert [item.competency for item in coverage] == expected
    assert len(coverage) == 5
    assert coverage[0].status == "covered"
    assert all(item.status == "missing" for item in coverage[1:])
    assert "Dependency injection" not in {item.competency for item in coverage}
    assert "ALL five are mandatory" in captured["user"]


def test_golden_fixtures_define_branch_and_all_five_statuses():
    golden = json.loads(Path("tests/golden.json").read_text(encoding="utf-8"))
    rubric = {chunk.competency for chunk in retrieval.rubric_chunks()}

    assert {item["expected_probe"] for item in golden} == {True, False}
    for item in golden:
        assert set(item["expected_coverage"]) == rubric
        expected_non_missing = {
            name for name, status in item["expected_coverage"].items() if status != "missing"
        }
        assert set(item["expected_competencies"]) == expected_non_missing


def test_semantic_index_contains_only_supplemental_reference_docs():
    chunks = retrieval._build_chunks()
    assert len(chunks) == 5
    assert all(chunk.doc_id != "00_scoring_rubric.md" for chunk in chunks)
    assert {chunk.competency for chunk in chunks} >= {
        "Dependency injection", "API design & security"
    }
