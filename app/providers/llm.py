import json
import time
from functools import lru_cache
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core import config
from app.core.obs import log_event
from app.providers import clients
from app.providers.errors import ProviderOutputError, ProviderUnavailableError

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=None)
def _strict_schema(model_cls: type[BaseModel]) -> dict:
    """Pydantic's model_json_schema() into the shape Groq's strict json_schema mode requires:
    every object needs additionalProperties=False and every property listed as required.

    Cached per model class: the schema is a pure function of the class, and there are four
    of them for the whole application. Callers must treat the result as read-only."""
    schema = model_cls.model_json_schema()

    def enforce(node: object) -> None:
        if isinstance(node, dict):
            if "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for value in node.values():
                enforce(value)
        elif isinstance(node, list):
            for item in node:
                enforce(item)

    enforce(schema)
    return schema


def _usage_fields(response: object) -> dict:
    """Token accounting for the provider.call log line. gpt-oss counts reasoning tokens
    inside completion_tokens, so both numbers are needed to size max_completion_tokens:
    the reasoning share is what actually moves when reasoning_effort changes.

    Defensive throughout — a provider response is not ours to trust, and the fakes in
    tests/test_provider_output.py carry no usage block at all."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        # Groq's prompt cache is automatic and exempt from rate limits, which are the binding
        # constraint here — so cache hit rate is an operational signal, not a cost curiosity.
        "cached_tokens": getattr(prompt_details, "cached_tokens", None),
    }


def complete(system: str, user: str, model_cls: type[T], thread_id: str = "-",
             stage: str | None = None) -> T:
    """One structured call to Groq's OpenAI-compatible chat.completions endpoint.
    One inline retry on a schema/parse failure (PLAN.md §12) — not a retry framework;
    transport-level retries (429/5xx) belong to the SDK and are configured in clients.py.

    `stage` names the call site so config.LLM_STAGES can carry a measured reasoning_effort
    and max_completion_tokens per task rather than one global setting — see config."""
    try:
        client = clients.groq()
    except Exception as exc:
        raise ProviderUnavailableError("groq", "evaluation") from exc
    profile = config.LLM_STAGES.get(stage or "", config.LLM_STAGE_FALLBACK)
    schema = _strict_schema(model_cls)
    # Groq's GPT-OSS guidance recommends putting task instructions in the user
    # message. Keeping the prompt text in prompts/ while composing one provider
    # message here improves adherence without coupling judge.py to Groq.
    messages: list[dict] = [{
        "role": "user",
        "content": f"{system}\n\nReturn only JSON matching the supplied schema.\n\nTask input:\n{user}",
    }]
    last_error: Exception | None = None
    for _ in range(2):
        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,
                temperature=config.GROQ_TEMPERATURE,
                top_p=config.GROQ_TOP_P,
                seed=config.GROQ_SEED,
                reasoning_effort=profile["reasoning_effort"],
                max_completion_tokens=profile["max_completion_tokens"],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": model_cls.__name__, "schema": schema, "strict": True},
                },
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) in (400, 422):
                raise ProviderOutputError("groq", "evaluation") from exc
            raise ProviderUnavailableError("groq", "evaluation") from exc
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        log_event("provider.call", thread_id, provider="groq", model=config.GROQ_MODEL,
                   kind="llm", stage=stage, reasoning_effort=profile["reasoning_effort"],
                   finish_reason=finish_reason,
                   latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                   **_usage_fields(response))
        if finish_reason == "length":
            # Strict decoding plus a truncated body means the JSON cannot be complete. Say so
            # explicitly: the parse error below would otherwise blame the model for what is a
            # max_completion_tokens ceiling set too low for this stage.
            log_event("provider.truncated", thread_id, provider="groq", model=config.GROQ_MODEL,
                      stage=stage, max_completion_tokens=profile["max_completion_tokens"])
        raw = choice.message.content or ""
        try:
            return model_cls.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as e:
            last_error = e
            log_event("provider.invalid_output", thread_id, provider="groq",
                      model=config.GROQ_MODEL, schema=model_cls.__name__, exc=type(e).__name__)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"That reply failed schema validation: {e}. "
                                                          f"Return only valid JSON matching the schema."})
    raise ProviderOutputError("groq", "evaluation") from last_error
