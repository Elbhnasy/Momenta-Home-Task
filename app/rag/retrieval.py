import re
from pathlib import Path

import numpy as np

from app.core import config
from app.core.obs import log_event
from app.providers import embeddings
from app.models.schemas import Chunk

_CORPUS_DIR = Path("data/corpus")

# Reference notes are labelled with the rubric competency they support (or "distractor" —
# no matching rubric competency). Hand-authored: over a fixed 6-file corpus this is data,
# not logic. TODO(tradeoff): derive from headings if the corpus grows.
_REFERENCE_COMPETENCY = {
    "01_async_concurrency.md": "Async & concurrency",
    "02_dependency_injection.md": "Dependency injection",       # distractor
    "03_data_access_ef_core.md": "Data access (EF Core)",
    "04_api_design_security.md": "API design & security",       # distractor
    "05_performance_memory.md": "Caching & performance",
}

_COMPETENCY_RE = re.compile(r"^## Competency: (.+)$", re.MULTILINE)


def _chunk_rubric(text: str) -> list[Chunk]:
    """Splits at '## Competency:' boundaries only — the '## Levels' block is not a
    competency and is deliberately never a chunk (PLAN.md §6.2); it's loaded verbatim
    elsewhere and always injected into the scoring prompt, not retrieved."""
    matches = list(_COMPETENCY_RE.finditer(text))
    chunks = []
    for i, m in enumerate(matches):
        competency = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        chunks.append(Chunk(doc_id="00_scoring_rubric.md", competency=competency,
                             text=f"{competency}\n\n{body}", score=0.0))
    return chunks


def _chunk_reference(doc_id: str, text: str) -> Chunk:
    competency = _REFERENCE_COMPETENCY[doc_id]
    body = text.split("\n", 1)[1].strip() if text.startswith("#") else text.strip()
    return Chunk(doc_id=doc_id, competency=competency, text=f"{competency}\n\n{body}", score=0.0)


def _build_chunks() -> list[Chunk]:
    chunks = []
    for doc_id in _REFERENCE_COMPETENCY:
        text = (_CORPUS_DIR / doc_id).read_text(encoding="utf-8")
        chunks.append(_chunk_reference(doc_id, text))
    return chunks


def rubric_chunks() -> list[Chunk]:
    """Return the complete, ordered scoring scope.

    Retrieval may add useful reference material, but it must never decide which of the
    question's five rubric competencies can be marked missing.
    """
    rubric_text = (_CORPUS_DIR / "00_scoring_rubric.md").read_text(encoding="utf-8")
    return _chunk_rubric(rubric_text)


_CHUNKS: list[Chunk] | None = None
_UNIT_EMBEDDINGS: np.ndarray | None = None  # L2-normalised once, so cosine is one matmul


def _get_index() -> tuple[list[Chunk], np.ndarray]:
    """Built once, lazily (one embedding call for the 5 supplemental chunks). Call
    warmup() at startup so the cost doesn't land on the first candidate's request."""
    global _CHUNKS, _UNIT_EMBEDDINGS
    if _CHUNKS is None:
        chunks = _build_chunks()
        vectors = embeddings.embed([c.text for c in chunks])
        _UNIT_EMBEDDINGS = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        _CHUNKS = chunks
    return _CHUNKS, _UNIT_EMBEDDINGS


def warmup() -> None:
    """Load the ONNX embedder and embed the corpus off the request path."""
    _get_index()


def _top_by_competency(chunks: list[Chunk], scores: np.ndarray, k: int) -> list[Chunk]:
    """Best-scoring supplemental chunk for each of the top-k competencies."""
    picked: list[Chunk] = []
    seen: set[str] = set()
    for i in np.argsort(-scores):
        chunk = chunks[i]
        if chunk.competency in seen:
            continue
        seen.add(chunk.competency)
        picked.append(chunk.model_copy(update={"score": float(scores[i])}))
        if len(picked) == k:
            break
    return picked


def retrieve(claims_summary_en: str, thread_id: str = "-") -> list[Chunk]:
    """Cosine similarity over five supplemental docs, including two distractors."""
    chunks, unit_embeddings = _get_index()
    query = embeddings.embed_query(claims_summary_en, thread_id=thread_id)
    scores = unit_embeddings @ (query / np.linalg.norm(query))
    top = _top_by_competency(chunks, scores, config.RETRIEVAL_K)
    log_event("retrieval.done", thread_id, doc_ids=[c.doc_id for c in top],
              competencies=[c.competency for c in top],
              scores=[round(c.score, 4) for c in top], k=config.RETRIEVAL_K)
    return top
