from __future__ import annotations

from types import SimpleNamespace as NS

import anthropic
import pytest
from pydantic import BaseModel

from app.services import ai, claude, gemini
from app.services.gemini import GeminiError, GroundedText


class _Log(list):
    fake = None


class Out(BaseModel):
    value: str = ""


@pytest.fixture
def calls(monkeypatch):
    log = _Log()

    def fake(name, fail=False):
        async def run(**kwargs):
            log.append(f"{name}:{kwargs['model']}")
            if fail:
                raise GeminiError(f"{name} quota reached")
            return Out(value=name)
        return run

    monkeypatch.setattr(gemini, "generate_json", fake("gemini"))
    monkeypatch.setattr(claude, "generate_json", fake("claude"))
    log.fake = fake
    return log


async def ask():
    return await ai.generate_json(api_key="", model="", prompt="p", schema=Out)


# --------------------------------------------------------------------------- routing
@pytest.mark.asyncio
async def test_no_provider_configured(calls):
    assert not ai.ai_available()
    with pytest.raises(GeminiError, match="ANTHROPIC_API_KEY"):
        await ask()


@pytest.mark.asyncio
async def test_only_claude_key_uses_claude_with_default_model(calls, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert ai.providers() == ["anthropic"] and ai.ai_available()
    assert (await ask()).value == "claude"
    assert calls == ["claude:claude-opus-5-5"]
    assert ai.describe() == "Claude (claude-opus-5-5; claude-haiku-5-5 for scoring)"


@pytest.mark.asyncio
async def test_provider_order_follows_ai_provider(calls, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert ai.providers() == ["gemini", "anthropic"]  # unset AI_PROVIDER: Gemini first when its key exists
    monkeypatch.setenv("AI_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    assert ai.providers() == ["anthropic", "gemini"]
    assert (await ask()).value == "claude" and calls == ["claude:claude-sonnet-5-5"]
    assert ai.describe("g", "gemini-x") == "Claude (claude-sonnet-5-5; claude-haiku-5-5 for scoring), fallback Gemini (gemini-x)"


@pytest.mark.asyncio
async def test_fast_tasks_use_haiku_writing_uses_opus(calls, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    await ai.generate_json(api_key="", model="", prompt="p", schema=Out, fast=True)
    await ai.generate_json(api_key="", model="", prompt="p", schema=Out)
    monkeypatch.setenv("ANTHROPIC_FAST_MODEL", "claude-sonnet-5-5")
    await ai.generate_json(api_key="", model="", prompt="p", schema=Out, fast=True)
    assert calls == ["claude:claude-haiku-5-5", "claude:claude-opus-5-5", "claude:claude-sonnet-5-5"]


@pytest.mark.asyncio
async def test_scoring_and_resume_extraction_request_fast_model(monkeypatch):
    from app.services import resume, scoring

    seen = []

    async def spy(**kwargs):
        seen.append(kwargs.get("fast", False))
        raise GeminiError("stop here")

    monkeypatch.setattr(scoring, "generate_json", spy)
    monkeypatch.setattr(resume, "generate_json", spy)
    from tests.test_scoring import PROFILE, GOOD
    await scoring.score_jobs([GOOD], PROFILE, gemini_api_key="k", gemini_model="m")
    with pytest.raises(GeminiError):
        await resume.analyze_gemini("resume text", "k", "m")
    assert seen == [True, True]


@pytest.mark.asyncio
async def test_falls_back_to_claude_when_gemini_fails(calls, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(gemini, "generate_json", calls.fake("gemini", fail=True))
    assert (await ask()).value == "claude"
    assert [c.split(":")[0] for c in calls] == ["gemini", "claude"]


@pytest.mark.asyncio
async def test_both_fail_reports_both(calls, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(gemini, "generate_json", calls.fake("gemini", fail=True))
    monkeypatch.setattr(claude, "generate_json", calls.fake("claude", fail=True))
    with pytest.raises(GeminiError, match="gemini quota reached; claude quota reached"):
        await ask()


@pytest.mark.asyncio
async def test_grounded_routes_too(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    async def fake(**kwargs):
        return GroundedText(text="answer", sources=[])

    monkeypatch.setattr(claude, "generate_grounded", fake)
    assert (await ai.generate_grounded(api_key="", model="", prompt="p")).text == "answer"


# --------------------------------------------------------------------------- Claude backend
def _status_error(cls, code: int, message: str = "boom"):
    return cls(message, response=NS(status_code=code, headers={}, request=None), body=None)


@pytest.mark.parametrize("exc, kind, text", [
    (_status_error(anthropic.AuthenticationError, 401), "auth", "key rejected"),
    (_status_error(anthropic.RateLimitError, 429), "quota", "rate limit"),
    (_status_error(anthropic.NotFoundError, 404), "unavailable", "claude-x"),
    (_status_error(anthropic.InternalServerError, 529), "overloaded", "overloaded"),
    (_status_error(anthropic.BadRequestError, 400, "Your credit balance is too low"), "quota", "no credit"),
])
def test_claude_error_mapping(exc, kind, text):
    err = claude._wrap(exc, "claude-x")
    assert err.kind == kind and text in str(err) and isinstance(err, GeminiError)


class _FakeMessages:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    async def parse(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def _fake_client(monkeypatch, responses) -> _FakeMessages:
    messages = _FakeMessages(responses)
    monkeypatch.setattr(claude, "_client", lambda key: NS(beta=NS(messages=messages)))
    return messages


@pytest.mark.asyncio
async def test_claude_generate_json(monkeypatch):
    fake = _fake_client(monkeypatch, [NS(stop_reason="end_turn", parsed_output=Out(value="ok")),
                                      NS(stop_reason="refusal", parsed_output=None)])
    out = await claude.generate_json(api_key="k", model="claude-opus-5-5", prompt="p", schema=Out)
    assert out.value == "ok"
    req = fake.requests[0]
    assert req["output_format"] is Out and req["output_config"] == {"effort": "medium"}
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in req
    with pytest.raises(GeminiError, match="declined"):
        await claude.generate_json(api_key="k", model="claude-haiku-5-5", prompt="p", schema=Out)
    assert "fallbacks" not in fake.requests[1]  # Haiku has no server-side refusal fallback


@pytest.mark.asyncio
async def test_claude_grounded_collects_citations_and_resumes_pause(monkeypatch):
    search = NS(type="web_search_tool_result", content=[NS(url="https://a.com", title="A"),
                                                          NS(url="https://b.com", title="B")])
    first = NS(stop_reason="pause_turn", content=[NS(type="text", text="Let me search.", citations=None),
                                                  NS(type="server_tool_use"), search])
    cited = NS(type="text", text='"rating": 4.1}', citations=[NS(url="https://b.com", title="B reviews")])
    second = NS(stop_reason="end_turn", content=[NS(type="text", text='{"known": true, ', citations=None), cited])
    fake = _fake_client(monkeypatch, [first, second])

    out = await claude.generate_grounded(api_key="k", model="claude-opus-5-5", prompt="Tell me")
    assert out.text == '{"known": true, "rating": 4.1}'  # only text after the last search
    assert [(s.url, s.title) for s in out.sources] == [("https://b.com", "B reviews")]  # cited pages win
    assert fake.requests[0]["tools"][0]["type"] == "web_search_20260209"
    resumed = fake.requests[1]["messages"]
    assert resumed[0]["content"] == "Tell me" and resumed[1]["role"] == "assistant"
