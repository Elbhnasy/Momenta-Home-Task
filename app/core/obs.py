import itertools
import json
import logging
import sys
import time
from collections import defaultdict
from functools import wraps

from langgraph.errors import GraphInterrupt

log = logging.getLogger("screener")

_BANNED = {
    "text", "transcript", "turns", "answer", "evidence", "audio",
    "api_key", "authorization", "prompt",
}

_counter: dict[str, itertools.count] = defaultdict(itertools.count)


def setup() -> None:
    # Arabic on a cp1252 Windows console otherwise raises UnicodeEncodeError inside the logger.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def log_event(event: str, thread_id: str, **fields) -> None:
    leaked = _BANNED & fields.keys()
    if leaked:
        raise ValueError(f"refusing to log sensitive fields: {leaked}")
    log.info(json.dumps({
        "event": event,
        "thread_id": thread_id,
        "seq": next(_counter[thread_id]),
        **fields,
    }, ensure_ascii=False))


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def traced_node(fn):
    @wraps(fn)
    def inner(state):
        t0 = time.perf_counter()
        try:
            out = fn(state)
        except GraphInterrupt:
            # Not a failure: interrupt() raises this to suspend the graph for a probe.
            raise
        except Exception as e:
            log_event("node.error", state["thread_id"], node=fn.__name__,
                      exc=type(e).__name__, elapsed_ms=_ms(t0))
            raise
        log_event("node.exit", state["thread_id"], node=fn.__name__, elapsed_ms=_ms(t0))
        return out
    return inner
