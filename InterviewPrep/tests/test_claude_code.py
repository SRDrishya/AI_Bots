import json

import pytest

from app.models import CategoryPlan, GeneratedQuestions
from app.services import ai, claude_code


def test_json_schema_inlines_nested_models():
    schema = claude_code.json_schema(GeneratedQuestions)
    assert "$defs" not in json.dumps(schema) and "$ref" not in json.dumps(schema)
    item = schema["properties"]["questions"]["items"]
    assert set(item["required"]) == {"question", "category", "level", "commonly_asked"}


def test_child_env_drops_api_keys_so_the_subscription_is_used(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    env = claude_code.child_env()
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env and "PATH" in env


def test_command_is_lean_and_tool_free():
    args = claude_code.command("claude", {"type": "object"}, model="sonnet")
    joined = " ".join(args)
    assert args[1] == "-p" and "--json-schema" in args
    assert args[args.index("--tools") + 1] == ""                 # no tools at all
    for flag in ("--no-session-persistence", "--strict-mcp-config", "--system-prompt"):
        assert flag in args
    assert "--bare" not in joined                                # --bare would ignore the Claude sign-in
    assert args[-2:] == ["--model", "sonnet"]


def test_find_executable_uses_configured_path(monkeypatch, tmp_path):
    exe = tmp_path / "claude.exe"
    exe.write_text("")
    monkeypatch.setenv("CLAUDE_CODE_PATH", str(exe))
    assert claude_code.find_executable() == str(exe)
    monkeypatch.setenv("CLAUDE_CODE_PATH", str(tmp_path / "missing.exe"))
    assert claude_code.find_executable() is None


def test_parse_output_success_and_errors():
    ok = json.dumps({"type": "result", "subtype": "success", "is_error": False,
                     "structured_output": {"categories": [{"name": "Async", "focus": "f"}]}})
    assert claude_code.parse_output(ok, CategoryPlan).categories[0].name == "Async"
    limit = json.dumps({"type": "result", "subtype": "success", "is_error": True,
                        "result": "Claude AI usage limit reached|1764000000"})
    with pytest.raises(claude_code.ClaudeCodeError, match="usage limit"):
        claude_code.parse_output(limit, CategoryPlan)
    with pytest.raises(claude_code.ClaudeCodeError, match="unreadable"):
        claude_code.parse_output("not json", CategoryPlan)


async def test_generate_json_sends_prompt_on_stdin(monkeypatch, tmp_path):
    exe = tmp_path / "claude.exe"
    exe.write_text("")
    monkeypatch.setenv("CLAUDE_CODE_PATH", str(exe))
    seen = {}

    class FakeProc:
        async def communicate(self, data):
            seen["stdin"] = data.decode()
            out = {"subtype": "success", "is_error": False, "structured_output": {"categories": []}}
            return json.dumps(out).encode(), b""

        def kill(self):
            pass

    async def fake_exec(*args, **kwargs):
        seen["args"], seen["env"] = args, kwargs["env"]
        return FakeProc()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setattr(claude_code.asyncio, "create_subprocess_exec", fake_exec)
    result = await claude_code.generate_json(prompt="a very long prompt " * 3000, schema=CategoryPlan)
    assert result.categories == []
    assert seen["stdin"].startswith("a very long prompt")      # not on the command line (Windows length limit)
    assert "ANTHROPIC_API_KEY" not in seen["env"]


async def test_router_uses_claude_code_mode(monkeypatch, tmp_path):
    exe = tmp_path / "claude.exe"
    exe.write_text("")
    monkeypatch.setenv("CLAUDE_CODE_PATH", str(exe))

    async def fake_generate(*, prompt, schema, model=""):
        return schema(categories=[])

    monkeypatch.setattr(claude_code, "generate_json", fake_generate)
    ai.set_mode("claude_code")
    try:
        assert ai.providers() == ["claude_code"] and ai.uses_claude_code()
        assert (await ai.generate_json(prompt="p", schema=CategoryPlan)).categories == []
        with pytest.raises(ai.AIError, match="Web search isn't used"):
            await ai.generate_grounded(prompt="p")
        monkeypatch.setenv("CLAUDE_CODE_PATH", str(tmp_path / "gone.exe"))
        assert not ai.ai_available()
    finally:
        ai.set_mode("auto")
