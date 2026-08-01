import io

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from app.api import routes
from app.models.schemas import Transcript
from app.providers import stt
from app.providers.errors import ProviderUnavailableError


def _upload(filename: str, content_type: str, payload: bytes = b"audio") -> UploadFile:
    return UploadFile(
        file=io.BytesIO(payload),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


@pytest.mark.parametrize("filename,content_type", [
    ("answer.mp3", "audio/mpeg"),
    ("recording.webm", "audio/webm;codecs=opus"),
    ("recording.mp4", "audio/mp4"),
    ("safari.mp4", "video/mp4"),
])
def test_supported_uploads_preserve_identity(filename, content_type):
    payload, actual_filename, actual_content_type = routes._read_audio_upload(
        _upload(filename, content_type)
    )
    assert payload == b"audio"
    assert actual_filename == filename
    assert actual_content_type == content_type


def test_upload_rejects_type_extension_mismatch():
    with pytest.raises(HTTPException) as exc:
        routes._read_audio_upload(_upload("answer.mp3", "audio/webm"))
    assert exc.value.status_code == 415


def test_upload_rejects_empty_and_oversized(monkeypatch):
    with pytest.raises(HTTPException) as empty:
        routes._read_audio_upload(_upload("answer.mp3", "audio/mpeg", b""))
    assert empty.value.status_code == 422

    monkeypatch.setattr(routes.config, "MAX_AUDIO_BYTES", 3)
    with pytest.raises(HTTPException) as large:
        routes._read_audio_upload(_upload("answer.mp3", "audio/mpeg", b"1234"))
    assert large.value.status_code == 413


def test_stt_passes_filename_and_mime_to_sdk(monkeypatch):
    captured = {}

    class SpeechToText:
        @staticmethod
        def convert(**kwargs):
            captured["name"] = kwargs["file"].name
            captured["content_type"] = kwargs["file"].content_type
            return type("Response", (), {"text": "plain English answer"})()

    client = type("Client", (), {"speech_to_text": SpeechToText()})()
    monkeypatch.setattr(stt.clients, "elevenlabs", lambda: client)

    result = stt.transcribe(b"bytes", "recording.webm", "audio/webm;codecs=opus")

    assert result.language == "en"
    assert captured == {
        "name": "recording.webm",
        "content_type": "audio/webm;codecs=opus",
    }


def test_start_maps_stt_provider_failure_to_useful_http_error(monkeypatch):
    def fail(*args, **kwargs):
        raise ProviderUnavailableError("elevenlabs", "transcription")

    monkeypatch.setattr(routes.stt, "transcribe", fail)
    with pytest.raises(HTTPException) as exc:
        routes.screening_start(_upload("answer.mp3", "audio/mpeg"))
    assert exc.value.status_code == 503
    assert "transcription service" in exc.value.detail.lower()


def test_start_passes_upload_identity_to_stt(monkeypatch):
    captured = {}

    def transcribe(payload, filename, content_type, thread_id):
        captured.update(filename=filename, content_type=content_type, payload=payload)
        return Transcript(text="answer", language="en")

    result = {
        "language": "en",
        "claims_summary_en": "answer",
        "retrieved": [],
        "coverage": [],
        "decision": {"probe": False, "target": None, "reason": "done"},
        "followups_used": 0,
        "score": 1,
        "justification": "Too vague.",
        "followup_question": None,
    }
    monkeypatch.setattr(routes.stt, "transcribe", transcribe)
    monkeypatch.setattr(routes.graph, "invoke", lambda *args, **kwargs: result)
    monkeypatch.setattr(routes.tts, "speak", lambda *args, **kwargs: b"ID3audio")

    response = routes.screening_start(_upload("safari.mp4", "audio/mp4", b"mp4"))

    assert response.status == "complete"
    assert captured == {"filename": "safari.mp4", "content_type": "audio/mp4", "payload": b"mp4"}


def test_finished_turn_degrades_to_text_when_tts_fails(monkeypatch):
    monkeypatch.setattr(routes.tts, "speak", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError()))
    assert routes._speak_or_degrade("verdict", "en", "thread") == ""
