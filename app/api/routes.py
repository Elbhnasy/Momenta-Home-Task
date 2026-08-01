import base64
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from langgraph.types import Command

from app.core import obs
from app.core import config
from app.core.question import load_question
from app.graph.build import graph
from app.models.schemas import Evaluation, TurnResponse
from app.providers import stt, tts
from app.providers.errors import ProviderError

router = APIRouter()

_AUDIO_TYPES = {
    "audio/mpeg": {".mp3"},
    "audio/mp3": {".mp3"},
    "audio/webm": {".webm"},
    "audio/mp4": {".mp4", ".m4a"},
    "audio/x-m4a": {".m4a", ".mp4"},
    "video/mp4": {".mp4"},  # Safari may label an audio-only MP4 container this way.
}


@router.get("/")
def index():
    return FileResponse(Path("static/index.html"))


@lru_cache(maxsize=1)
def _question_audio_b64() -> str:
    """The question never changes, so it is spoken once per process, not once per visitor."""
    return base64.b64encode(tts.speak(load_question(), language="en")).decode()


@router.get("/api/question")
def get_question():
    try:
        return {"question": load_question(), "audio_b64": _question_audio_b64()}
    except ProviderError as exc:
        raise _provider_http_error(exc, "-") from exc


def _read_audio_upload(audio: UploadFile) -> tuple[bytes, str, str]:
    filename = Path(audio.filename or "").name
    content_type = audio.content_type or ""
    media_type = content_type.partition(";")[0].strip().lower()
    suffix = Path(filename).suffix.lower()

    if media_type not in _AUDIO_TYPES or suffix not in _AUDIO_TYPES[media_type]:
        raise HTTPException(415, "Unsupported audio type. Upload MP3, WebM, MP4, or M4A audio.")
    payload = audio.file.read(config.MAX_AUDIO_BYTES + 1)
    if not payload:
        raise HTTPException(422, "The uploaded audio file is empty.")
    if len(payload) > config.MAX_AUDIO_BYTES:
        raise HTTPException(413, f"Audio exceeds the {config.MAX_AUDIO_BYTES // (1024 * 1024)} MB limit.")
    return payload, filename, content_type


def _provider_http_error(exc: ProviderError, thread_id: str) -> HTTPException:
    cause = type(exc.__cause__).__name__ if exc.__cause__ else type(exc).__name__
    obs.log_event("provider.failed", thread_id, provider=exc.provider,
                  kind=exc.operation, exc=cause)
    return HTTPException(exc.status_code, exc.public_message)


def _speak_or_degrade(text: str, language: str, thread_id: str) -> str:
    """A finished evaluation must survive a TTS outage. The verdict is already in the JSON
    body and the page renders it as text, so returning silent audio costs the turn its
    voice; raising here would cost the candidate the whole interview and the graph state
    that produced it."""
    try:
        return base64.b64encode(tts.speak(text, language, thread_id=thread_id)).decode()
    except Exception as e:
        obs.log_event("tts.failed", thread_id, exc=type(e).__name__)
        return ""


def _finish_turn(result: dict, thread_id: str, transcript_text: str) -> TurnResponse:
    status = "awaiting_followup" if "__interrupt__" in result else "complete"
    evaluation = Evaluation.from_state(result)
    spoken_text = evaluation.spoken_text()
    audio_b64 = _speak_or_degrade(spoken_text, result["language"], thread_id)
    if status == "complete":
        obs.log_event("session.end", thread_id, score=result.get("score"),
                      followups_used=result["followups_used"])
    return TurnResponse(
        status=status,
        thread_id=thread_id,
        transcript=transcript_text,
        evaluation=evaluation,
        spoken_text=spoken_text,
        audio_b64=audio_b64,
    )


@router.post("/api/screening/start")
def screening_start(audio: UploadFile):
    thread_id = uuid.uuid4().hex
    payload, filename, content_type = _read_audio_upload(audio)
    try:
        transcript = stt.transcribe(payload, filename, content_type, thread_id=thread_id)
    except ProviderError as exc:
        raise _provider_http_error(exc, thread_id) from exc
    if not transcript.text.strip():
        raise HTTPException(422, "Could not transcribe any speech from the recording — please try again.")

    obs.log_event("session.start", thread_id, language=transcript.language, source="mic")
    config_ = {"configurable": {"thread_id": thread_id}}
    try:
        result = graph.invoke(
            {"thread_id": thread_id, "language": transcript.language, "turns": [transcript.text],
             "followups_used": 0, "followup_question": None},
            config_,
        )
    except ProviderError as exc:
        raise _provider_http_error(exc, thread_id) from exc
    return _finish_turn(result, thread_id, transcript.text)


@router.post("/api/screening/{thread_id}/resume")
def screening_resume(thread_id: str, audio: UploadFile):
    config_ = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config_)
    if not state.values:
        raise HTTPException(404, "session not found — start a new screening (state is in-process)")
    if not state.next:
        raise HTTPException(409, "session is not awaiting a follow-up")

    payload, filename, content_type = _read_audio_upload(audio)
    try:
        transcript = stt.transcribe(payload, filename, content_type, thread_id=thread_id)
    except ProviderError as exc:
        raise _provider_http_error(exc, thread_id) from exc
    if not transcript.text.strip():
        raise HTTPException(422, "Could not transcribe any speech from the recording — please try again.")

    obs.log_event("interrupt.resumed", thread_id, turn_index=len(state.values["turns"]),
                  answer_chars=len(transcript.text))
    try:
        result = graph.invoke(Command(resume=transcript.text), config_)
    except ProviderError as exc:
        raise _provider_http_error(exc, thread_id) from exc
    return _finish_turn(result, thread_id, transcript.text)
