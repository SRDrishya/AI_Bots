"""Claude Code (the `claude` CLI) as an AI provider.

Uses the Claude subscription you are signed in to in Claude Code instead of API credits. Each call runs
`claude -p` headless with structured JSON output and is kept lean: no tools, no MCP servers, no
settings/hooks, no saved session, and a short system prompt instead of Claude Code's large default
one (about 20x less usage per call). ANTHROPIC_API_KEY is removed from the child environment so Claude
Code uses your sign-in, not the API account.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from app.services.gemini import GeminiError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

TIMEOUT = 600.0   # seconds; a batch of detailed answers can take a few minutes
MAX_TURNS = 4
SYSTEM_PROMPT = ("You are an expert technical interviewer and software architect. Follow the instructions "
                 "exactly and return only the requested structured output.")
# Without these, Claude Code would use the API key from .env (which may have no credit) instead of the
# subscription you are signed in with.
_STRIPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


class ClaudeCodeError(GeminiError):
    """Claude Code failure; message is safe to show the user."""


def _version(path: Path) -> tuple[int, ...]:
    match = re.search(r"claude-code-(\d+(?:\.\d+)*)", str(path))
    return tuple(int(p) for p in match.group(1).split(".")) if match else (0,)


def find_executable() -> str | None:
    """CLAUDE_CODE_PATH, then `claude` on PATH, then the copy bundled with the VS Code / Cursor extension."""
    configured = os.getenv("CLAUDE_CODE_PATH", "").strip()
    if configured:
        return configured if Path(configured).exists() else None
    for name in ("claude", "claude.exe"):
        found = shutil.which(name)
        if found:
            return found
    home = Path.home()
    candidates = [home / ".local" / "bin" / "claude.exe", home / ".local" / "bin" / "claude"]
    for editor in (".vscode", ".cursor", ".vscode-insiders"):
        bundled = (home / editor / "extensions").glob("anthropic.claude-code-*/resources/native-binary/claude*")
        candidates += sorted(bundled, key=_version, reverse=True)
    return next((str(p) for p in candidates if p.is_file()), None)


def json_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema with every $ref inlined (one self-contained schema for --json-schema)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs[node["$ref"].split("/")[-1]])
            return {k: inline(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    return inline(schema)


def command(executable: str, schema: dict, model: str = "") -> list[str]:
    args = [executable, "-p", "--output-format", "json", "--json-schema", json.dumps(schema),
            "--max-turns", str(MAX_TURNS), "--tools", "", "--no-session-persistence", "--strict-mcp-config",
            "--setting-sources", "", "--system-prompt", SYSTEM_PROMPT]
    return [*args, "--model", model] if model else args


def child_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}


def parse_output(stdout: str, schema: type[T]) -> T:
    try:
        data = json.loads(stdout)
    except ValueError as exc:
        raise ClaudeCodeError(f"Claude Code returned unreadable output: {stdout[:160]}") from exc
    if data.get("is_error") or data.get("subtype") not in (None, "success"):
        detail = str(data.get("result") or data.get("subtype") or "unknown error")
        if "limit" in detail.lower():
            raise ClaudeCodeError(f"Claude Code usage limit reached: {detail[:200]}")
        raise ClaudeCodeError(f"Claude Code error: {detail[:200]}")
    payload = data.get("structured_output")
    if payload is None:
        try:
            payload = json.loads(data.get("result") or "")
        except ValueError as exc:
            raise ClaudeCodeError("Claude Code did not return the expected JSON") from exc
    try:
        return schema.model_validate(payload)
    except ValueError as exc:
        raise ClaudeCodeError("Claude Code returned JSON that didn't match the expected format") from exc


async def generate_json(*, prompt: str, schema: type[T], model: str = "") -> T:
    executable = find_executable()
    if not executable:
        raise ClaudeCodeError("Claude Code was not found — install it, or set CLAUDE_CODE_PATH in .env")
    workdir = tempfile.mkdtemp(prefix="interviewprep-cc-")  # empty folder: nothing for it to read
    try:
        proc = await asyncio.create_subprocess_exec(
            *command(executable, json_schema(schema), model), cwd=workdir, env=child_env(),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            # the prompt goes through stdin: Windows limits command lines to ~32k characters
            stdout, stderr = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), TIMEOUT)
        except TimeoutError as exc:
            proc.kill()
            raise ClaudeCodeError("Claude Code timed out") from exc
    except OSError as exc:
        raise ClaudeCodeError(f"Could not start Claude Code: {exc}") from exc
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    out = stdout.decode("utf-8", "replace").strip()
    if not out:
        raise ClaudeCodeError(f"Claude Code failed: {stderr.decode('utf-8', 'replace').strip()[:200] or 'no output'}")
    return parse_output(out, schema)
