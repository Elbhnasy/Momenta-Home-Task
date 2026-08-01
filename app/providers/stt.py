import io
import re
import time

from app.core import config
from app.core.obs import log_event
from app.models.schemas import Transcript
from app.providers import clients
from app.providers.errors import ProviderUnavailableError

_ARABIC_RE = re.compile("[؀-ۿ]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _derive_language(text: str) -> str:
    """Arabic-script ratio over letters, PLAN.md §7.3. No STT API returns a
    code-mixed label directly, so this is computed rather than requested."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    arabic_ratio = sum(1 for c in letters if _ARABIC_RE.match(c)) / len(letters)
    has_latin_tokens = bool(_LATIN_WORD_RE.search(text))
    if arabic_ratio >= 0.60:
        return "ar-en" if has_latin_tokens else "ar"
    if arabic_ratio < 0.10:
        return "en"
    return "ar-en"


def transcribe(audio: bytes, filename: str, content_type: str, thread_id: str = "-") -> Transcript:
    """Scribe auto-detects the spoken language when language_code is omitted —
    never pin it (PLAN.md §7.2/CLAUDE.md §7): pinning mangles code-mixed audio."""
    try:
        client = clients.elevenlabs()
        buf = io.BytesIO(audio)
        # ElevenLabs uses the file object's metadata to identify the container. Preserve
        # what FastAPI received so MP3, browser WebM, and Safari MP4 are not mislabeled.
        buf.name = filename
        buf.content_type = content_type

        t0 = time.perf_counter()
        response = client.speech_to_text.convert(
            model_id=config.ELEVENLABS_STT_MODEL,
            file=buf,
            keyterms=config.STT_VOCAB_HINT,
        )
    except Exception as exc:
        raise ProviderUnavailableError("elevenlabs", "transcription") from exc
    log_event("provider.call", thread_id, provider="elevenlabs", model=config.ELEVENLABS_STT_MODEL,
              kind="stt", latency_ms=round((time.perf_counter() - t0) * 1000, 1))

    text = (response.text or "").strip()
    return Transcript(text=text, language=_derive_language(text))
