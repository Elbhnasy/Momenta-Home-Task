from pathlib import Path

from app.graph import build
from app.models.schemas import Chunk, CoverageItem


def test_graph_nodes_emit_plain_checkpoint_values(monkeypatch):
    chunk = Chunk(doc_id="doc.md", competency="Diagnostic method", text="criteria", score=0.5)
    monkeypatch.setattr(build.retrieval, "retrieve", lambda *args: [chunk])
    retrieved = build.retrieve_node({"claims_summary_en": "summary", "thread_id": "test"})
    assert retrieved == {"retrieved": [chunk.model_dump(mode="json")]}
    assert isinstance(retrieved["retrieved"][0], dict)

    coverage = [CoverageItem(
        competency="Diagnostic method", status="covered", evidence="measured"
    )]
    monkeypatch.setattr(build.judge, "analyze_coverage", lambda *args: coverage)
    analyzed = build.analyze_node({
        "thread_id": "test",
        "turns": ["measured"],
        "retrieved": retrieved["retrieved"],
        "language": "en",
        "followups_used": 0,
    })
    assert isinstance(analyzed["coverage"][0], dict)
    assert isinstance(analyzed["decision"], dict)


def test_static_ui_never_injects_model_output_as_html():
    html = Path("static/index.html").read_text(encoding="utf-8")
    assert ".innerHTML" not in html
    assert "document.createElement(\"td\")" in html
    assert "cell.textContent" in html
    assert "replaceChildren()" in html
