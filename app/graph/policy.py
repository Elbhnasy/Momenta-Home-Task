from app.models.schemas import CoverageItem, Decision


def decide(coverage: list[CoverageItem], followups_used: int) -> Decision:
    if followups_used >= 1:                      # hard cap, in code
        return Decision(probe=False, target=None, reason="follow-up budget spent")
    gaps = [c for c in coverage if c.status in ("partial", "missing")]
    if not gaps:
        return Decision(probe=False, target=None, reason="all competencies covered — score now")
    if len([c for c in coverage if c.status == "missing"]) >= 2:
        return Decision(probe=False, target=None,
                         reason="weak across the board; one probe cannot move it")
    if len(gaps) == 1:
        return Decision(probe=True, target=gaps[0].competency,
                         reason=f"single gap in {gaps[0].competency}; a probe may move the score")
    return Decision(probe=False, target=None, reason="diffuse gaps; score now")
