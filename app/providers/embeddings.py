import time

import numpy as np
from fastembed import TextEmbedding

from app.core import config
from app.core.obs import log_event

_embedder: TextEmbedding | None = None


def _get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=config.EMBEDDING_MODEL)
    return _embedder


def embed(texts: list[str], thread_id: str = "-") -> np.ndarray:
    """Embeds corpus passages (not queries — use embed_query for those)."""
    t0 = time.perf_counter()
    vectors = list(_get_embedder().passage_embed(texts))
    log_event("provider.call", thread_id, provider="fastembed", model=config.EMBEDDING_MODEL,
               kind="embed", latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    return np.array(vectors)


def embed_query(text: str, thread_id: str = "-") -> np.ndarray:
    t0 = time.perf_counter()
    vector = next(iter(_get_embedder().query_embed(text)))
    log_event("provider.call", thread_id, provider="fastembed", model=config.EMBEDDING_MODEL,
               kind="embed", latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    return np.array(vector)
