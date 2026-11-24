"""Provider router: Gemini or Anthropic Claude, with automatic fallback to the other.

AI_PROVIDER in .env picks the main provider. If it fails (quota, overload, bad key,
model gone, malformed output) and the other provider has a key, the same request is
retried there. Keys and models all come from .env.
"""
from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import BaseModel

from app.config import get_env_settings
from app.services import claude, gemini
from app.services.gemini import GeminiError, GroundedText

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
AIError = GeminiError  # any provider failure; message is safe to show the user

AI_ORDER = ("gemini", "anthropic")
NO_KEY_HINT = "a Gemini or Claude API key (GEMINI_API_KEY or ANTHROPIC_API_KEY in .env)"


def providers() -> list[str]:
    """Usable providers, main one first."""
    env = get_env_settings()
    have = {"gemini": bool(env.gemini_api_key), "anthropic": bool(env.anthropic_api_key)}
    order = [env.ai_provider, *(p for p in AI_ORDER if p != env.ai_provider)]
    return [p for p in order if have[p]]



def ai_available() -> bool:
    return bool(providers())


def describe() -> str:
    """e.g. 'Gemini (gemini-3.8-flash), fallback Claude (claude-opus-5-5)' for the status bar."""
    env = get_env_settings()
    label = {"gemini": f"Gemini ({env.gemini_model})", "anthropic": f"Claude ({env.anthropic_model})"}
    return ", fallback ".join(label[p] for p in providers())


async def _route(call_gemini, call_claude, fast: bool = False):
    env = get_env_settings()
    chain = providers()
    if not chain:
        raise AIError(f"No AI provider configured — add {NO_KEY_HINT}")
    errors: list[str] = []
    for provider in chain:
        try:
            if provider == "gemini":
                return await call_gemini(env)
            return await call_claude(env, env.anthropic_fast_model if fast else env.anthropic_model)
        except GeminiError as exc:
            errors.append(str(exc))
            if provider != chain[-1]:
                log.warning("%s failed (%s); falling back to the other AI provider", provider, exc)
    raise AIError("; ".join(errors))


async def generate_json(*, prompt: str, schema: type[T], temperature: float = 0.3, fast: bool = False) -> T:
    """Structured JSON from the main AI provider, falling back to the other one.
    fast=True: cheap extraction/scoring; on Claude it uses ANTHROPIC_FAST_MODEL."""
    return await _route(
        lambda env: gemini.generate_json(api_key=env.gemini_api_key, model=env.gemini_model, prompt=prompt,
                                         schema=schema, temperature=temperature),
        lambda env, model: claude.generate_json(api_key=env.anthropic_api_key, model=model, prompt=prompt,
                                                schema=schema),
        fast,
    )


async def generate_grounded(*, prompt: str, temperature: float = 0.3) -> GroundedText:
    """Answer backed by live web search (Google Search on Gemini, web search on Claude)."""
    return await _route(
        lambda env: gemini.generate_grounded(api_key=env.gemini_api_key, model=env.gemini_model, prompt=prompt,
                                             temperature=temperature),
        lambda env, model: claude.generate_grounded(api_key=env.anthropic_api_key, model=model, prompt=prompt),
    )
