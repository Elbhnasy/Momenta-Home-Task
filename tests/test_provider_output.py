import pytest
from pydantic import BaseModel

from app.providers import llm
from app.providers.errors import ProviderOutputError, ProviderUnavailableError


class _Reply(BaseModel):
    value: int


def test_malformed_structured_output_retries_then_fails_clearly(monkeypatch):
    message = type("Message", (), {"content": "not-json"})()
    choice = type("Choice", (), {"message": message})()
    response = type("Response", (), {"choices": [choice]})()

    class Completions:
        calls = 0

        @classmethod
        def create(cls, **kwargs):
            cls.calls += 1
            return response

    client = type("Client", (), {
        "chat": type("Chat", (), {"completions": Completions()})()
    })()
    monkeypatch.setattr(llm.clients, "groq", lambda: client)

    with pytest.raises(ProviderOutputError, match="invalid response"):
        llm.complete("system", "user", _Reply)
    assert Completions.calls == 2


def test_provider_call_failure_is_sanitized(monkeypatch):
    class Completions:
        @staticmethod
        def create(**kwargs):
            raise RuntimeError("secret provider detail")

    client = type("Client", (), {
        "chat": type("Chat", (), {"completions": Completions()})()
    })()
    monkeypatch.setattr(llm.clients, "groq", lambda: client)

    with pytest.raises(ProviderUnavailableError) as exc:
        llm.complete("system", "user", _Reply)
    assert "secret provider detail" not in str(exc.value)


def test_provider_400_is_classified_as_invalid_output(monkeypatch):
    class BadOutput(Exception):
        status_code = 400

    class Completions:
        @staticmethod
        def create(**kwargs):
            raise BadOutput("schema generation failed")

    client = type("Client", (), {
        "chat": type("Chat", (), {"completions": Completions()})()
    })()
    monkeypatch.setattr(llm.clients, "groq", lambda: client)

    with pytest.raises(ProviderOutputError):
        llm.complete("system", "user", _Reply)
