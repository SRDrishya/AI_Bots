from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.services import gemini
from app.services.gemini import GeminiError, generate_json


class Out(BaseModel):
    city: str


def install_fake_client(monkeypatch, behaviour: dict[str, list]):
    """behaviour[model] = list of outcomes per call: Exception instance or Out."""
    calls: list[str] = []

    async def generate_content(*, model, contents, config):
        calls.append(model)
        outcome = behaviour[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(parsed=outcome, text=outcome.model_dump_json())

    fake = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake)
    monkeypatch.setattr(gemini, "RETRY_BACKOFF", 0)
    gemini.reset_cooldowns()
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "fallback-a,fallback-b")
    return calls


async def call(model="main"):
    return await generate_json(api_key="k", model=model, prompt="p", schema=Out)


async def test_success_first_try(monkeypatch):
    calls = install_fake_client(monkeypatch, {"main": [Out(city="Chennai")]})
    assert (await call()).city == "Chennai"
    assert calls == ["main"]


async def test_overloaded_retries_once_then_falls_back(monkeypatch):
    overloaded = Exception("503 UNAVAILABLE. This model is currently experiencing high demand.")
    calls = install_fake_client(monkeypatch, {
        "main": [overloaded, overloaded],
        "fallback-a": [Out(city="Chennai")],
    })
    assert (await call()).city == "Chennai"
    assert calls == ["main", "main", "fallback-a"]


async def test_retired_model_skips_straight_to_fallback(monkeypatch):
    calls = install_fake_client(monkeypatch, {
        "main": [Exception("404 NOT_FOUND. models/main is no longer available to new users")],
        "fallback-a": [Exception("400 INVALID_ARGUMENT weird")],
        "fallback-b": [Out(city="Chennai")],
    })
    assert (await call()).city == "Chennai"
    assert calls == ["main", "fallback-a", "fallback-b"]


@pytest.mark.parametrize(
    ("error", "message"),
    [("400 API key not valid. Please pass a valid API key.", "Gemini API key rejected"),
     ("429 RESOURCE_EXHAUSTED quota exceeded", "Gemini quota / rate limit reached")],
)
async def test_fatal_errors_stop_immediately(monkeypatch, error, message):
    calls = install_fake_client(monkeypatch, {"main": [Exception(error)]})
    with pytest.raises(GeminiError, match=message):
        await call()
    assert calls == ["main"]


async def test_all_models_fail_reports_last_error(monkeypatch):
    install_fake_client(monkeypatch, {
        "main": [Exception("404 NOT_FOUND")],
        "fallback-a": [Exception("503 UNAVAILABLE"), Exception("503 UNAVAILABLE")],
        "fallback-b": [Exception("503 UNAVAILABLE"), Exception("503 UNAVAILABLE")],
    })
    with pytest.raises(GeminiError, match=r"overloaded right now \(fallback-b\)"):
        await call()


async def test_failed_model_is_skipped_on_next_call(monkeypatch):
    overloaded = Exception("503 UNAVAILABLE")
    calls = install_fake_client(monkeypatch, {
        "main": [overloaded, overloaded],
        "fallback-a": [Out(city="A"), Out(city="B")],
    })
    assert (await call()).city == "A"
    assert (await call()).city == "B"
    assert calls == ["main", "main", "fallback-a", "fallback-a"]  # 2nd call skips cooling "main"


def test_all_models_cooling_still_tries_them(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "a")
    gemini.reset_cooldowns()
    gemini._cool("main", 60)
    gemini._cool("a", 60)
    assert gemini.model_chain("main") == ["main", "a"]
    gemini.reset_cooldowns()


def test_model_chain_dedupes(monkeypatch):
    gemini.reset_cooldowns()
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "a, main ,b")
    assert gemini.model_chain("main") == ["main", "a", "b"]


def test_thinking_config_per_family():
    assert gemini._thinking_config("gemini-2.5-flash").thinking_budget == 0
    assert gemini._thinking_config("gemini-3.8-flash").thinking_level.value.lower() == "low"
    assert gemini._thinking_config("gemini-3.5-flash-lite") is None
