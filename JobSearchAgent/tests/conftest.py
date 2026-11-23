from __future__ import annotations

import pytest

API_KEY_VARS = ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "AI_PROVIDER", "ADZUNA_APP_ID", "ADZUNA_APP_KEY", "RAPIDAPI_KEY", "SERPAPI_KEY")


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch):
    """Tests must never use (or spend quota on) the developer's real .env keys."""
    monkeypatch.setattr("app.config.load_dotenv", lambda *args, **kwargs: False)
    for var in API_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
