"""Provider client construction: one place for credentials, timeouts and connection reuse.

Each SDK client owns an HTTP connection pool, so a client built per call would re-handshake
TLS on every provider hop. STT and TTS share one ElevenLabs client for the same reason —
same host, same credential, no reason for two pools.

The ElevenLabs client keeps the SDK's default retry count: its failures here are auth or
quota, which no amount of retrying fixes. The Groq client does not — on a free-tier 8K
tokens/minute bucket, 429 is an expected steady-state condition during a gate run rather
than an anomaly, and two retries cannot span a bucket refill. See config.LLM_MAX_RETRIES.
"""
from functools import lru_cache

from elevenlabs.client import ElevenLabs
from openai import OpenAI

from app.core import config


def _require(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


@lru_cache(maxsize=1)
def groq() -> OpenAI:
    """Groq's OpenAI-compatible endpoint, via the OpenAI SDK."""
    return OpenAI(
        api_key=_require(config.GROQ_API_KEY, "GROQ_API_KEY"),
        base_url=config.GROQ_BASE_URL,
        timeout=config.LLM_TIMEOUT_S,
        max_retries=config.LLM_MAX_RETRIES,
    )


@lru_cache(maxsize=1)
def elevenlabs() -> ElevenLabs:
    """Shared by stt.py (Scribe) and tts.py (multilingual v2)."""
    return ElevenLabs(
        api_key=_require(config.ELEVENLABS_API_KEY, "ELEVENLABS_API_KEY"),
        timeout=config.ELEVENLABS_TIMEOUT_S,
    )
