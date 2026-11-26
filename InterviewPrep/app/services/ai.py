"""Provider router: Gemini, the Claude API, or Claude Code.

The provider mode is chosen in Settings:
  auto         Gemini first, the Claude API as fallback (AI_PROVIDER in .env sets which goes first)
  gemini       Gemini only
  anthropic    Claude API only
  claude_code  Claude Code (your Claude subscription). Web search isn't used in this mode, so the
               builder has Claude Code write the questions (marked AI-generated) as well as the answers.
"""
from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import BaseModel

from app.config import get_env_settings
from app.services import claude, claude_code, gemini
from app.services.gemini import GeminiError, GroundedText

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
AIError = GeminiError  # any provider failure; message is safe to show the user

AI_ORDER = ("gemini", "anthropic")
MODES = {
    "auto": "Auto: Gemini, then Claude API",
    "gemini": "Gemini only",
    "anthropic": "Claude API only",
    "claude_code": "Claude Code (your Claude subscription)",
}
NO_KEY_HINT = "a Gemini or Claude API key (GEMINI_API_KEY or ANTHROPIC_API_KEY in .env), or choose Claude Code in Settings"

_mode = "auto"


def set_mode(mode: str) -> None:
    global _mode
    _mode = mode if mode in MODES else "auto"


def mode() -> str:
    return _mode


def uses_claude_code() -> bool:
    return _mode == "claude_code"


def providers() -> list[str]:
    """Usable providers for the current mode, main one first."""
    if _mode == "claude_code":
        return ["claude_code"] if claude_code.find_executable() else []
    env = get_env_settings()
    have = {"gemini": bool(env.gemini_api_key), "anthropic": bool(env.anthropic_api_key)}
    order = [env.ai_provider, *(p for p in AI_ORDER if p != env.ai_provider)]
    if _mode in AI_ORDER:
        order = [_mode]
    return [p for p in order if have[p]]


def ai_available() -> bool:
    return bool(providers())


def describe() -> str:
    """e.g. 'Gemini (gemini-3.8-flash), fallback Claude (claude-opus-5-5)' for the status bar."""
    env = get_env_settings()
    label = {"gemini": f"Gemini ({env.gemini_model})", "anthropic": f"Claude API ({env.anthropic_model})",
             "claude_code": "Claude Code (your Claude subscription)"}
    return ", fallback ".join(label[p] for p in providers())


async def _route(call_gemini, call_claude, call_claude_code, fast: bool = False):
    env = get_env_settings()
    chain = providers()
    if not chain:
        if _mode == "claude_code":
            raise AIError("Claude Code was not found — install it, or set CLAUDE_CODE_PATH in .env")
        raise AIError(f"No AI provider configured — add {NO_KEY_HINT}")
    errors: list[str] = []
    for provider in chain:
        try:
            if provider == "gemini":
                return await call_gemini(env)
            if provider == "claude_code":
                return await call_claude_code(env)
            return await call_claude(env, env.anthropic_fast_model if fast else env.anthropic_model)
        except GeminiError as exc:
            errors.append(str(exc))
            if provider != chain[-1]:
                log.warning("%s failed (%s); falling back to the other AI provider", provider, exc)
    raise AIError("; ".join(errors))


async def generate_json(*, prompt: str, schema: type[T], temperature: float = 0.3, fast: bool = False) -> T:
    """Structured JSON from the current provider(s).
    fast=True: cheap extraction/scoring; on the Claude API it uses ANTHROPIC_FAST_MODEL."""
    return await _route(
        lambda env: gemini.generate_json(api_key=env.gemini_api_key, model=env.gemini_model, prompt=prompt,
                                         schema=schema, temperature=temperature),
        lambda env, model: claude.generate_json(api_key=env.anthropic_api_key, model=model, prompt=prompt,
                                                schema=schema),
        lambda env: claude_code.generate_json(prompt=prompt, schema=schema, model=env.claude_code_model),
        fast,
    )


async def generate_grounded(*, prompt: str, temperature: float = 0.3) -> GroundedText:
    """Answer backed by live web search (Google Search on Gemini, web search on the Claude API)."""
    async def no_web(env):
        raise AIError("Web search isn't used in Claude Code mode")

    return await _route(
        lambda env: gemini.generate_grounded(api_key=env.gemini_api_key, model=env.gemini_model, prompt=prompt,
                                             temperature=temperature),
        lambda env, model: claude.generate_grounded(api_key=env.anthropic_api_key, model=model, prompt=prompt),
        no_web,
    )
