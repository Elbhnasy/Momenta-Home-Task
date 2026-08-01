from fastapi import APIRouter

from app.core import config

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "groq_key_present": bool(config.GROQ_API_KEY),
        "elevenlabs_key_present": bool(config.ELEVENLABS_API_KEY),
    }
