from functools import lru_cache
from pathlib import Path

_QUESTION_PATH = Path("QUESTION.md")


@lru_cache(maxsize=1)
def load_question() -> str:
    text = _QUESTION_PATH.read_text(encoding="utf-8")
    quoted_lines = [line.removeprefix(">").strip() for line in text.splitlines() if line.startswith(">")]
    return " ".join(quoted_lines).strip('"')
