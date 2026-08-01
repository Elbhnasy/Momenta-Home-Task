import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.question import load_question
from app.core.obs import log_event
from app.providers import llm
from app.models.schemas import Chunk, CoverageItem
from app.rag import retrieval

_RUBRIC_PATH = Path("data/corpus/00_scoring_rubric.md")
_PROMPTS_DIR = Path("prompts")


@lru_cache(maxsize=1)
def _load_levels_block() -> str:
    text = _RUBRIC_PATH.read_text(encoding="utf-8")
    match = re.search(r"^## Levels\n(.*?)(?=^## )", text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip()


@lru_cache(maxsize=None)
def _load_prompt(name: str) -> str:
    """CLAUDE.md 'Prompt Rules': prompts live in prompts/, never hardcoded in Python."""
    return (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


# --- structured LLM response wrappers (wire format only, not API contracts — those live in schemas.py) ---

class _ClaimsSummary(BaseModel):
    claims_summary_en: str


class _CoverageReport(BaseModel):
    items: list[CoverageItem]


class _Followup(BaseModel):
    question: str


class _Verdict(BaseModel):
    score: int = Field(strict=True, ge=1, le=5)
    justification: str


def _chunks_text(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.competency} | {c.doc_id} | score={c.score:.3f}]\n{c.text}" for c in chunks)


def summarize(turns: list[str], followup_question: str | None, thread_id: str) -> str:
    parts = [f'Initial answer: "{turns[0]}"']
    if followup_question and len(turns) > 1:
        parts.append(f'Follow-up question asked: "{followup_question}"')
        parts.append(f'Follow-up answer: "{turns[1]}"')
    user = f"Interview question: {load_question()}\n\n" + "\n".join(parts)
    result = llm.complete(_load_prompt("summarize"), user, _ClaimsSummary, thread_id=thread_id,
                          stage="summarize")
    return result.claims_summary_en


def analyze_coverage(turns: list[str], retrieved: list[Chunk], language: str, thread_id: str) -> list[CoverageItem]:
    mandatory_rubric = retrieval.rubric_chunks()
    requested = [c.competency for c in mandatory_rubric]
    transcript = " ".join(turns)
    user = (
        f"Interview question: {load_question()}\n\n"
        f"Transcript (candidate's own words, language={language}): \"{transcript}\"\n\n"
        f"Competencies to judge (ALL five are mandatory): {requested}\n\n"
        f"Mandatory rubric criteria:\n{_chunks_text(mandatory_rubric)}\n\n"
        f"Semantically retrieved supplemental material:\n{_chunks_text(retrieved)}\n\n"
        "Final calibration: inspecting the endpoint's actual queries for N+1 or indexes is "
        "evidence gathering and therefore partial Diagnostic method even when the answer lacks "
        "a full APM/measure/verify sequence."
    )
    result = llm.complete(_load_prompt("analyze_coverage"), user, _CoverageReport, thread_id=thread_id,
                          stage="analyze_coverage")

    by_competency = {item.competency: item for item in result.items if item.competency in requested}
    coverage = [
        by_competency.get(c, CoverageItem(competency=c, status="missing", evidence=None))
        for c in requested
    ]

    unverified = sum(
        1 for c in coverage
        if c.status in ("covered", "partial") and (not c.evidence or c.evidence not in transcript)
    )
    log_event("coverage.done", thread_id, statuses={c.competency: c.status for c in coverage},
               evidence_unverified=unverified)
    return coverage


def write_followup(target: str, retrieved: list[Chunk], language: str, thread_id: str) -> str:
    rubric_chunk = next((c for c in retrieval.rubric_chunks() if c.competency == target),
                         next((c for c in retrieved if c.competency == target), None))
    rubric_text = rubric_chunk.text if rubric_chunk else target
    user = f"Language code: {language}\n\nGap to probe: {target}\n\nRubric guidance:\n{rubric_text}"
    result = llm.complete(_load_prompt("write_followup"), user, _Followup, thread_id=thread_id,
                          stage="write_followup")
    return result.question


def score(turns: list[str], coverage: list[CoverageItem], retrieved: list[Chunk], language: str,
          thread_id: str) -> tuple[int, str]:
    transcript = " ".join(turns)
    coverage_text = "\n".join(f"- {c.competency}: {c.status}" + (f' (evidence: "{c.evidence}")' if c.evidence else "")
                               for c in coverage)
    user = (
        f"Interview question: {load_question()}\n\n"
        f"Transcript (language={language}): \"{transcript}\"\n\n"
        f"Coverage map:\n{coverage_text}\n\n"
        f"Rubric levels:\n{_load_levels_block()}\n\n"
        f"Mandatory rubric competency reference:\n{_chunks_text(retrieval.rubric_chunks())}\n\n"
        f"Semantically retrieved supplemental material:\n{_chunks_text(retrieved)}"
    )
    result = llm.complete(_load_prompt("score"), user, _Verdict, thread_id=thread_id, stage="score")
    return result.score, result.justification
