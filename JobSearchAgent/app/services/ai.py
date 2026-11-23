"""Provider router: Gemini or Anthropic Claude, with automatic fallback to the other.

AI_PROVIDER in .env picks the main provider. If it fails (quota / rate limit, overload,
bad key, model gone, malformed output) and the other provider has a key, the same request
is retried there. Callers keep passing the Gemini key/model they resolved; the Claude
key/models come from .env: ANTHROPIC_MODEL for writing (cover letters, tailored resumes,
company research) and ANTHROPIC_FAST_MODEL for high-volume extraction (fast=True: resume
parsing, job scoring).
"""
from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import BaseModel

from app.config import DEFAULT_GEMINI_MODEL, get_env_settings
from app.services import claude, gemini
from app.services.gemini import GeminiError, GroundedText

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
AIError = GeminiError  # any provider failure; message is safe to show the user

NO_KEY_HINT = "a Gemini or Claude API key (GEMINI_API_KEY or ANTHROPIC_API_KEY in .env)"


def providers(gemini_key: str = "") -> list[str]:
    """Usable providers, main one first."""
    env = get_env_settings()
    have = {"gemini": bool(gemini_key or env.gemini_api_key), "anthropic": bool(env.anthropic_api_key)}
    order = [env.ai_provider, *(p for p in ("gemini", "anthropic") if p != env.ai_provider)]
    return [p for p in order if have[p]]


def ai_available(gemini_key: str = "") -> bool:
    return bool(providers(gemini_key))


def describe(gemini_key: str = "", gemini_model: str = "") -> str:
    """e.g. 'Claude (claude-opus-5-5), fallback Gemini (gemini-3.8-flash)' for the status page."""
    env = get_env_settings()
    label = {"gemini": f"Gemini ({gemini_model or env.gemini_model})",
             "anthropic": f"Claude ({env.anthropic_model}; {env.anthropic_fast_model} for scoring)"}
    names = [label[p] for p in providers(gemini_key)]
    return ", fallback ".join(names) if names else ""


async def _route(gemini_key: str, call_gemini, call_claude, fast: bool = False):
    env = get_env_settings()
    chain = providers(gemini_key)
    if not chain:
        raise AIError(f"No AI provider configured — add {NO_KEY_HINT}")
    errors: list[str] = []
    for provider in chain:
        try:
            if provider == "gemini":
                return await call_gemini()
            return await call_claude(env.anthropic_api_key, env.anthropic_fast_model if fast else env.anthropic_model)
        except GeminiError as exc:
            errors.append(str(exc))
            if provider != chain[-1]:
                log.warning("%s failed (%s); falling back to the other AI provider", provider, exc)
    raise AIError("; ".join(errors))


async def generate_json(*, api_key: str, model: str, prompt: str, schema: type[T], temperature: float = 0.2,
                        fast: bool = False) -> T:
    """Structured JSON from the main AI provider, falling back to the other one.
    fast=True: high-volume extraction/scoring; on Claude it uses the cheaper ANTHROPIC_FAST_MODEL."""
    return await _route(
        api_key,
        lambda: gemini.generate_json(api_key=api_key or get_env_settings().gemini_api_key,
                                     model=model or DEFAULT_GEMINI_MODEL, prompt=prompt, schema=schema,
                                     temperature=temperature),
        lambda key, m: claude.generate_json(api_key=key, model=m, prompt=prompt, schema=schema),
        fast,
    )


async def generate_grounded(*, api_key: str, model: str, prompt: str, temperature: float = 0.2) -> GroundedText:
    """Answer backed by live web search (Google Search on Gemini, web search on Claude)."""
    return await _route(
        api_key,
        lambda: gemini.generate_grounded(api_key=api_key or get_env_settings().gemini_api_key,
                                         model=model or DEFAULT_GEMINI_MODEL, prompt=prompt,
                                         temperature=temperature),
        lambda key, m: claude.generate_grounded(api_key=key, model=m, prompt=prompt),
    )
