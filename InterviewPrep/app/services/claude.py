"""Anthropic Claude backend: structured JSON output and web-search-grounded answers.

Mirrors app.services.gemini so app.services.ai can use either provider. Errors are raised
as GeminiError (the app-wide "AI failed" error) with messages that are safe to show the user.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TypeVar

from pydantic import BaseModel

from app.services.gemini import GeminiError, GroundedText, WebSource

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

JSON_TIMEOUT = 240.0       # seconds; a batch of detailed answers can take a while
GROUNDED_TIMEOUT = 300.0   # web search runs several searches server-side
MAX_TOKENS = 16_000
MAX_SEARCHES = 5
MAX_CONTINUATIONS = 3      # resumes after stop_reason "pause_turn"
EFFORT = "medium"          # extraction / rewriting work; set explicitly (Opus 5.5 default is medium too)

# Models that accept server-side refusal fallbacks ("default" routing by refusal category).
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1"}


class ClaudeError(GeminiError):
    """Claude failure; `kind` tells the router whether another provider might succeed."""

    def __init__(self, message: str, kind: str = "other"):
        super().__init__(message)
        self.kind = kind  # auth | quota | overloaded | unavailable | other


@lru_cache(maxsize=4)
def _client(api_key: str):
    import anthropic

    return anthropic.AsyncAnthropic(api_key=api_key, max_retries=2)  # SDK retries 429 / 5xx itself


def _fallback_kwargs(model: str) -> dict:
    if model in _FALLBACK_MODELS:
        return {"betas": [_FALLBACK_BETA], "fallbacks": "default"}
    return {}


def _wrap(exc: Exception, model: str) -> ClaudeError:
    import anthropic

    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return ClaudeError("Claude API key rejected", "auth")
    if isinstance(exc, anthropic.NotFoundError):
        return ClaudeError(f"Claude model not available ({model}) — check ANTHROPIC_MODEL", "unavailable")
    if isinstance(exc, anthropic.RateLimitError):
        return ClaudeError("Claude rate limit reached", "quota")
    if isinstance(exc, anthropic.BadRequestError):
        message = getattr(exc, "message", str(exc))
        if "credit balance" in message.lower():
            return ClaudeError("Claude account has no credit left (console.anthropic.com → Billing)", "quota")
        return ClaudeError(f"Claude rejected the request: {message[:200]}", "other")
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code >= 500:
            return ClaudeError("Claude is overloaded right now", "overloaded")
        return ClaudeError(f"Claude error {exc.status_code}: {getattr(exc, 'message', '')[:200]}", "other")
    if isinstance(exc, anthropic.APITimeoutError):
        return ClaudeError("Claude timed out", "overloaded")
    if isinstance(exc, anthropic.APIConnectionError):
        return ClaudeError("Couldn't reach Claude (network error)", "overloaded")
    return ClaudeError(f"Claude error: {str(exc)[:200]}", "other")


def _check_stop(response) -> None:
    if response.stop_reason == "refusal":
        raise ClaudeError("Claude declined this request", "other")
    if response.stop_reason == "max_tokens":
        raise ClaudeError("Claude's answer was cut off (too long)", "other")


async def generate_json(*, api_key: str, model: str, prompt: str, schema: type[T], effort: str = EFFORT) -> T:
    """Ask Claude for JSON matching `schema` (structured outputs) and return it validated."""
    import anthropic

    try:
        response = await _client(api_key).beta.messages.parse(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            output_config={"effort": effort},
            timeout=JSON_TIMEOUT,
            **_fallback_kwargs(model),
        )
    except anthropic.AnthropicError as exc:
        raise _wrap(exc, model) from exc
    except ValueError as exc:  # output didn't validate against the schema
        raise ClaudeError("Claude returned JSON that didn't match the expected format") from exc
    _check_stop(response)
    parsed = response.parsed_output
    if not isinstance(parsed, schema):
        raise ClaudeError("Claude returned an empty response")
    return parsed


def _grounded_from(content) -> GroundedText:
    """Final answer text (after the last search) plus the web pages it cited (or searched)."""
    last_search = max((i for i, b in enumerate(content) if b.type == "web_search_tool_result"), default=-1)
    texts, cited, searched = [], {}, {}
    for i, block in enumerate(content):
        if block.type == "web_search_tool_result" and isinstance(block.content, list):  # error -> object
            for r in block.content:
                searched.setdefault(r.url, r.title or "source")
        elif block.type == "text":
            if i > last_search:
                texts.append(block.text)
            for c in getattr(block, "citations", None) or []:
                if getattr(c, "url", None):
                    cited.setdefault(c.url, getattr(c, "title", "") or "source")
    sources = cited or dict(list(searched.items())[:8])
    return GroundedText(text="".join(texts).strip(), sources=[WebSource(title=t, url=u) for u, t in sources.items()])


async def generate_grounded(*, api_key: str, model: str, prompt: str) -> GroundedText:
    """Claude answer backed by live web search, with the pages it used."""
    import anthropic

    messages: list[dict] = [{"role": "user", "content": prompt}]
    content: list = []
    try:
        for _ in range(MAX_CONTINUATIONS + 1):
            response = await _client(api_key).beta.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                messages=messages,
                tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": MAX_SEARCHES}],
                output_config={"effort": EFFORT},
                timeout=GROUNDED_TIMEOUT,
                **_fallback_kwargs(model),
            )
            content.extend(response.content)
            if response.stop_reason != "pause_turn":
                break
            # Server-side search loop paused: send the turn back unchanged and it resumes.
            messages = [messages[0], {"role": "assistant", "content": content}]
    except anthropic.AnthropicError as exc:
        raise _wrap(exc, model) from exc
    _check_stop(response)
    grounded = _grounded_from(content)
    if not grounded.text:
        raise ClaudeError("Claude returned an empty response")
    return grounded
