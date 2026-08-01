from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Transcript(BaseModel):
    text: str
    language: Literal["ar", "en", "ar-en"]


class Chunk(BaseModel):
    doc_id: str
    competency: str
    text: str
    score: float


class CoverageItem(BaseModel):
    competency: str
    status: Literal["covered", "partial", "missing"]
    evidence: str | None  # verbatim transcript quote; null only when status == "missing"

    @field_validator("evidence", mode="before")
    @classmethod
    def remove_wrapping_quote_marks(cls, value):
        if not isinstance(value, str):
            return value
        value = value.strip()
        quote_pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
        if len(value) >= 2 and quote_pairs.get(value[0]) == value[-1]:
            return value[1:-1].strip()
        return value

    @model_validator(mode="after")
    def validate_evidence(self) -> "CoverageItem":
        if self.status == "missing" and self.evidence is not None:
            raise ValueError("evidence must be null when status is missing")
        if self.status != "missing" and not (self.evidence or "").strip():
            raise ValueError("evidence must be non-empty when status is covered or partial")
        return self


class Decision(BaseModel):
    probe: bool
    target: str | None
    reason: str


class Evaluation(BaseModel):
    claims_summary_en: str
    retrieved: list[Chunk]
    coverage: list[CoverageItem]
    decision: Decision
    score: int | None = Field(default=None, strict=True, ge=1, le=5)
    justification: str | None
    followup_question: str | None

    def spoken_text(self) -> str:
        if self.score is None:
            return self.followup_question or ""
        return f"{self.score}/5 — {self.justification}"

    @classmethod
    def from_state(cls, state: dict) -> "Evaluation":
        return cls(
            claims_summary_en=state["claims_summary_en"],
            retrieved=state["retrieved"],
            coverage=state["coverage"],
            decision=state["decision"],
            score=state.get("score"),
            justification=state.get("justification"),
            followup_question=state.get("followup_question"),
        )


class TurnResponse(BaseModel):
    status: Literal["awaiting_followup", "complete"]
    thread_id: str
    transcript: str
    evaluation: Evaluation
    spoken_text: str
    audio_b64: str
