import time

from elevenlabs import VoiceSettings

from app.core import config
from app.core.obs import log_event
from app.providers import clients
from app.providers.errors import ProviderUnavailableError


def speak(text: str, language: str, thread_id: str = "-") -> bytes:
    """`language` isn't sent to the API: eleven_multilingual_v2 doesn't accept
    `language_code` (ElevenLabs docs: "not supported for multilingual_v2 models"),
    and unlike OpenAI TTS this SDK has no `instructions` steering parameter — the
    model reads language from the text itself. Kept in the signature so callers
    don't need a provider-specific branch, and because Phase 4's voice/pronunciation
    tuning may still need it."""
    try:
        client = clients.elevenlabs()
        t0 = time.perf_counter()
        chunks = client.text_to_speech.convert(
            voice_id=config.ELEVENLABS_VOICE_ID,
            text=text,
            model_id=config.ELEVENLABS_TTS_MODEL,
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75),
        )
        audio = b"".join(chunks)
    except Exception as exc:
        raise ProviderUnavailableError("elevenlabs", "speech synthesis") from exc
    log_event("provider.call", thread_id, provider="elevenlabs", model=config.ELEVENLABS_TTS_MODEL,
              kind="tts", latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    return audio
