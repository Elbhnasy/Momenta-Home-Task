import json
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.types import Command

from app.core import config as app_config, obs
from app.graph.build import graph
from app.providers import stt, tts
from app.providers.stt import _ARABIC_RE
from app.models.schemas import Evaluation
from app.rag import retrieval

logging.basicConfig(level=logging.INFO)
obs.setup()

SAMPLES_DIR = Path("data/audio")
ARTIFACTS_DIR = Path("artifacts")
GOLDEN = json.loads(Path("tests/golden.json").read_text(encoding="utf-8"))
NO_FOLLOWUP_SENTINEL = "I don't have anything more to add."
DISTRACTORS = {"02_dependency_injection.md", "04_api_design_security.md"}
# Groq's free 20B tier has an 8K-token rolling minute. Answer 1 takes the longer probe
# branch, so let that window clear before the next isolated sample instead of depending
# on a 429 retry race. This is gate pacing, not application latency.
INTER_SAMPLE_COOLDOWN_S = 55
RUBRIC_COMPETENCIES = {
    "Diagnostic method",
    "Data access (EF Core)",
    "Async & concurrency",
    "Caching & performance",
    "Tradeoff & communication",
}
PROVIDER_CONFIGURATION = {
    "stt": {"provider": "elevenlabs", "model": app_config.ELEVENLABS_STT_MODEL},
    "llm": {
        "provider": "groq",
        "model": app_config.GROQ_MODEL,
        "temperature": app_config.GROQ_TEMPERATURE,
        "top_p": app_config.GROQ_TOP_P,
        "seed": app_config.GROQ_SEED,
        "stages": app_config.LLM_STAGES,
    },
    "embeddings": {"provider": "fastembed-local", "model": app_config.EMBEDDING_MODEL},
    "tts": {"provider": "elevenlabs", "model": app_config.ELEVENLABS_TTS_MODEL},
}


class _CapturingHandler(logging.Handler):
    """Every screener log line is already the JSON string obs.log_event built. Parsing
    it back in memory lets assertions inspect every retrieval.done event in a session
    (a probing sample retrieves twice), not just the final state."""

    def __init__(self):
        super().__init__()
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(json.loads(record.getMessage()))


def _arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _ARABIC_RE.match(c)) / len(letters)


def _is_mp3(audio: bytes) -> bool:
    return audio.startswith(b"ID3") or (
        len(audio) >= 2 and audio[0] == 0xFF and (audio[1] & 0xE0) == 0xE0
    )


def _speak_and_check(text: str, language: str, thread_id: str, label: str,
                     failures: list[str]) -> dict:
    audio = tts.speak(text, language, thread_id=thread_id)
    valid = _is_mp3(audio)
    if not valid:
        failures.append(f"{label} TTS did not return valid MP3 data ({len(audio)} bytes)")
    return {"label": label, "language": language, "bytes": len(audio), "valid_mp3": valid}


def _run_one(sample_path: Path, golden: dict, capture: _CapturingHandler) -> dict:
    failures: list[str] = []
    t0 = time.perf_counter()
    thread_id = uuid.uuid4().hex
    transcript = stt.transcribe(
        sample_path.read_bytes(),
        filename=sample_path.name,
        content_type="audio/mpeg",
        thread_id=thread_id,
    )
    obs.log_event("session.start", thread_id, language=transcript.language, source=f"sample:{sample_path.name}")

    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"thread_id": thread_id, "language": transcript.language, "turns": [transcript.text],
         "followups_used": 0, "followup_question": None},
        config,
    )

    probed = "__interrupt__" in result
    initial_evaluation = Evaluation.from_state(result)
    followup_question = None
    spoken_outputs: list[dict] = []
    if probed:
        followup_question = result["followup_question"]
        if result.get("score") is not None:
            failures.append("score is not None on the interrupted (probing) turn's result")
        spoken_outputs.append(_speak_and_check(
            initial_evaluation.spoken_text(), result["language"], thread_id, "follow-up", failures
        ))
        obs.log_event("interrupt.resumed", thread_id, turn_index=len(result["turns"]),
                       answer_chars=len(NO_FOLLOWUP_SENTINEL))
        result = graph.invoke(Command(resume=NO_FOLLOWUP_SENTINEL), config)

    obs.log_event("session.end", thread_id, score=result.get("score"),
                   followups_used=result["followups_used"],
                   total_elapsed_ms=round((time.perf_counter() - t0) * 1000, 1))

    evaluation = Evaluation.from_state(result)
    spoken_outputs.append(_speak_and_check(
        evaluation.spoken_text(), result["language"], thread_id, "verdict", failures
    ))

    # 1. score range + justification
    score = result.get("score")
    if not (isinstance(score, int) and 1 <= score <= 5):
        failures.append(f"score {score!r} is not an int in 1..5")
    if not (result.get("justification") or "").strip():
        failures.append("justification is empty")

    # 2. distractors absent from every retrieve call in the session, not just the final one
    doc_ids_seen: set[str] = set()
    for rec in capture.records:
        if rec.get("event") == "retrieval.done" and rec.get("thread_id") == thread_id:
            doc_ids_seen.update(rec.get("doc_ids", []))
    hit_distractors = doc_ids_seen & DISTRACTORS
    if hit_distractors:
        failures.append(f"distractor doc(s) in top-k: {sorted(hit_distractors)}")

    # 3. spoken text language matches state["language"]
    language = result["language"]
    spoken_text = evaluation.spoken_text()
    ratio = _arabic_ratio(spoken_text)
    if language in ("ar", "ar-en") and ratio <= 0.30:
        failures.append(f"language={language} but spoken text Arabic ratio {ratio:.2f} <= 0.30")
    if language == "en" and ratio >= 0.05:
        failures.append(f"language=en but spoken text Arabic ratio {ratio:.2f} >= 0.05")

    # 4. decision.reason non-empty on both branches
    if not (evaluation.decision.reason or "").strip():
        failures.append("decision.reason is empty")

    # 5. expected branch and complete, stable five-competency coverage
    if probed != golden["expected_probe"]:
        failures.append(f"probe={probed}, expected {golden['expected_probe']}")

    actual_coverage = {item.competency: item.status for item in initial_evaluation.coverage}
    if set(actual_coverage) != RUBRIC_COMPETENCIES:
        missing = sorted(RUBRIC_COMPETENCIES - set(actual_coverage))
        extra = sorted(set(actual_coverage) - RUBRIC_COMPETENCIES)
        failures.append(f"coverage scope is not all five competencies; missing={missing}, extra={extra}")
    coverage_rank = {"missing": 0, "partial": 1, "covered": 2}
    coverage_mismatches = {}
    for competency, expected_status in golden["expected_coverage"].items():
        actual_status = actual_coverage.get(competency)
        if expected_status == "missing":
            matches = actual_status == "missing"
        else:
            matches = actual_status in coverage_rank and (
                coverage_rank[actual_status] >= coverage_rank[expected_status]
            )
        if not matches:
            coverage_mismatches[competency] = {
                "actual": actual_status,
                "minimum_expected": expected_status,
            }
    if coverage_mismatches:
        failures.append(f"coverage expectations failed: {coverage_mismatches}")

    actual_non_missing = {name for name, status in actual_coverage.items() if status != "missing"}
    expected_non_missing = set(golden["expected_competencies"])
    if actual_non_missing != expected_non_missing:
        failures.append(
            f"non-missing competencies {sorted(actual_non_missing)} != "
            f"expected {sorted(expected_non_missing)}"
        )

    final_scope = {item.competency for item in evaluation.coverage}
    if final_scope != RUBRIC_COMPETENCIES:
        failures.append("final post-probe coverage scope is not all five competencies")

    # 6. golden agreement within +/-1
    expected = golden["expected_score"]
    if isinstance(score, int) and abs(score - expected) > 1:
        failures.append(f"score {score} outside golden ±1 (expected {expected})")

    verdict = {
        "provider_configuration": PROVIDER_CONFIGURATION,
        "claims_summary_en": result["claims_summary_en"],
        "retrieved": [c.model_dump() for c in evaluation.retrieved],
        "coverage": [c.model_dump() for c in evaluation.coverage],
        "decision": evaluation.decision.model_dump(),
        "followup_question": followup_question,
        "score": score,
        "justification": result.get("justification"),
        "spoken_outputs": spoken_outputs,
    }
    return {
        "file": sample_path.name,
        "probed": probed,
        "score": score,
        "expected_score": expected,
        "competencies": len(evaluation.coverage),
        "distractors_clean": not hit_distractors,
        "verdict": verdict,
        "failures": failures,
    }


def _aborted(sample_path: Path, golden: dict, exc: Exception) -> dict:
    """A provider timing out on sample 2 should not cost us samples 1 and 3. The run is
    still a failure — it just reports which sample failed and keeps going."""
    return {
        "file": sample_path.name,
        "probed": False,
        "score": None,
        "expected_score": golden["expected_score"],
        "competencies": 0,
        "distractors_clean": True,
        "verdict": {
            "provider_configuration": PROVIDER_CONFIGURATION,
            "error": f"{type(exc).__name__}: {exc}",
        },
        "failures": [f"run aborted: {type(exc).__name__}: {exc}"],
    }


def _render_summary(results: list[dict]) -> str:
    lines = [
        f"Provider configuration: `{json.dumps(PROVIDER_CONFIGURATION, ensure_ascii=False)}`",
        "",
        "| Sample | Score | Golden Δ | Probed? | Competencies judged | Distractors clean | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        delta = "-" if r["score"] is None else r["score"] - r["expected_score"]
        status = "PASS" if not r["failures"] else f"FAIL ({len(r['failures'])})"
        lines.append(
            f"| {r['file']} | {r['score']} | {delta} | {'yes' if r['probed'] else 'no'} | "
            f"{r['competencies']}/5 | {'yes' if r['distractors_clean'] else 'no'} | {status} |"
        )
    total = len(results)
    clean = sum(1 for r in results if not r["failures"])
    lines.append("")
    lines.append(f"**{clean}/{total} samples passed all assertions.**")
    return "\n".join(lines) + "\n"


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    capture = _CapturingHandler()
    log_file = logging.FileHandler(ARTIFACTS_DIR / "run.log", mode="w", encoding="utf-8")
    log_file.setFormatter(logging.Formatter("%(message)s"))
    screener_logger = logging.getLogger("screener")
    screener_logger.addHandler(capture)
    screener_logger.addHandler(log_file)

    retrieval.warmup()  # same as the server's startup hook: keep the cold index off sample 1
    obs.log_event("index.ready", "-")

    results = []
    for index, golden in enumerate(GOLDEN):
        sample_path = SAMPLES_DIR / golden["file"]
        try:
            results.append(_run_one(sample_path, golden, capture))
        except Exception as e:
            results.append(_aborted(sample_path, golden, e))
        if index < len(GOLDEN) - 1:
            time.sleep(INTER_SAMPLE_COOLDOWN_S)

    for i, r in enumerate(results, start=1):
        (ARTIFACTS_DIR / f"eval_{i}.json").write_text(
            json.dumps(r["verdict"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
    (ARTIFACTS_DIR / "summary.md").write_text(_render_summary(results), encoding="utf-8")

    screener_logger.removeHandler(capture)
    screener_logger.removeHandler(log_file)
    log_file.close()

    all_failures = [(r["file"], f) for r in results for f in r["failures"]]
    for file, f in all_failures:
        print(f"FAIL [{file}]: {f}")
    clean = sum(1 for r in results if not r["failures"])
    print(f"{clean}/{len(results)} samples passed all assertions.")
    sys.exit(1 if all_failures else 0)


if __name__ == "__main__":
    main()
