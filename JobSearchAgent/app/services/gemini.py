"""Thin wrapper around google-genai for structured (JSON-schema) generation.

Resilience: Gemini models are frequently overloaded (503) or retired for new
users (404). Each call walks a model chain — the configured model, then
GEMINI_FALLBACK_MODELS — retrying a 503 once with a short backoff. Errors that
another model can't fix (bad key, exhausted quota) stop immediately.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from app.config import get_env_settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

ATTEMPT_TIMEOUT = 30.0
RETRY_BACKOFF = 2.0


class GeminiError(RuntimeError):
    """Any failure talking to Gemini; message is safe to show the user."""


class _Fatal(GeminiError):
    """Errors no other model will fix."""


def _classify(exc: Exception) -> tuple[str, str]:
    """-> (kind, message). kind: fatal | overloaded | unavailable | other"""
    text = str(exc)
    lowered = text.lower()
    if "api key not valid" in lowered or "api_key_invalid" in lowered or "permission_denied" in lowered:
        return "fatal", "Gemini API key rejected"
    if "resource_exhausted" in lowered or "429" in lowered or "quota" in lowered:
        return "fatal", "Gemini quota / rate limit reached"
    if "503" in lowered or "unavailable" in lowered or "overloaded" in lowered:
        return "overloaded", "Gemini is overloaded right now"
    if "404" in lowered or "not_found" in lowered or "no longer available" in lowered:
        return "unavailable", "Gemini model not available"
    return "other", f"Gemini error: {text[:200]}"


def _thinking_config(model: str):
    from google.genai import types

    # Extraction / ranking doesn't need deep reasoning; keep it fast and cheap.
    if model.startswith("gemini-2.5-flash"):
        return types.ThinkingConfig(thinking_budget=0)
    if model.startswith("gemini-3") and "lite" not in model:
        return types.ThinkingConfig(thinking_level="low")  # "minimal" is rejected by 3.x flash
    return None


# Circuit breaker: a model that just failed is skipped for a while, so later
# batches/searches go straight to a working fallback instead of waiting on it.
OVERLOAD_COOLDOWN = 600.0        # 10 min after 503 / timeout
UNAVAILABLE_COOLDOWN = 86_400.0  # 24 h after "model not available" (404)
_cooldown_until: dict[str, float] = {}


def _cool(model: str, seconds: float) -> None:
    _cooldown_until[model] = time.monotonic() + seconds


def reset_cooldowns() -> None:
    _cooldown_until.clear()


def model_chain(model: str) -> list[str]:
    fallbacks = get_env_settings().gemini_fallback_models
    chain = list(dict.fromkeys([m for m in [model, *fallbacks] if m]))
    now = time.monotonic()
    healthy = [m for m in chain if _cooldown_until.get(m, 0) <= now]
    return healthy or chain  # everything cooling down -> try them all anyway


async def _attempt(client, model: str, prompt: str, schema: type[T], temperature: float) -> T:
    from google.genai import types

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
        thinking_config=_thinking_config(model),
    )
    response = await asyncio.wait_for(
        client.aio.models.generate_content(model=model, contents=prompt, config=config),
        timeout=ATTEMPT_TIMEOUT,
    )
    parsed = response.parsed
    if isinstance(parsed, schema):
        return parsed
    if not response.text:
        raise GeminiError("Gemini returned an empty response")
    try:
        return schema.model_validate_json(response.text)
    except ValueError as exc:
        raise GeminiError("Gemini returned malformed JSON") from exc


@dataclass
class WebSource:
    title: str
    url: str


@dataclass
class GroundedText:
    text: str
    sources: list[WebSource]


async def _attempt_grounded(client, model: str, prompt: str, temperature: float) -> GroundedText:
    from google.genai import types

    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=temperature,
        thinking_config=_thinking_config(model),
    )
    response = await asyncio.wait_for(
        client.aio.models.generate_content(model=model, contents=prompt, config=config),
        timeout=ATTEMPT_TIMEOUT,
    )
    if not response.text:
        raise GeminiError("Gemini returned an empty response")
    sources: list[WebSource] = []
    candidates = getattr(response, "candidates", None) or []
    metadata = getattr(candidates[0], "grounding_metadata", None) if candidates else None
    for chunk in (getattr(metadata, "grounding_chunks", None) or []):
        web = getattr(chunk, "web", None)
        if web and getattr(web, "uri", None):
            sources.append(WebSource(title=getattr(web, "title", "") or "source", url=web.uri))
    return GroundedText(text=response.text, sources=sources)


async def generate_grounded(*, api_key: str, model: str, prompt: str, temperature: float = 0.2) -> GroundedText:
    """Gemini answer backed by live Google Search, with the web sources it used.
    Raises GeminiError if search grounding isn't available on this key/plan."""
    from google import genai

    client = genai.Client(api_key=api_key)
    return await _run_chain(model, lambda m: _attempt_grounded(client, m, prompt, temperature))


async def generate_json(
    *,
    api_key: str,
    model: str,
    prompt: str,
    schema: type[T],
    temperature: float = 0.2,
) -> T:
    """Ask Gemini for JSON matching `schema` and return it parsed."""
    from google import genai

    client = genai.Client(api_key=api_key)
    return await _run_chain(model, lambda m: _attempt(client, m, prompt, schema, temperature))


async def _run_chain(model: str, call):
    """Try `call(model)` across the model chain with retry / fallback / cooldown rules."""
    last_error = "Gemini error"
    for candidate in model_chain(model):
        for attempt in range(2):
            try:
                return await call(candidate)
            except GeminiError as exc:  # malformed / empty output: try next model
                last_error = str(exc)
                break
            except asyncio.TimeoutError:
                last_error = f"Gemini timed out ({candidate})"
                _cool(candidate, OVERLOAD_COOLDOWN)
                break
            except Exception as exc:
                kind, message = _classify(exc)
                last_error = f"{message} ({candidate})" if kind != "fatal" else message
                if kind == "fatal":
                    raise _Fatal(message) from exc
                if kind == "overloaded" and attempt == 0:
                    await asyncio.sleep(RETRY_BACKOFF)
                    continue
                if kind == "overloaded":
                    _cool(candidate, OVERLOAD_COOLDOWN)
                elif kind == "unavailable":
                    _cool(candidate, UNAVAILABLE_COOLDOWN)
                break
        log.info("Gemini model %s failed (%s); trying next", candidate, last_error)
    raise GeminiError(last_error)
