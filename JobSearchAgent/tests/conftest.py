from __future__ import annotations

import pytest

API_KEY_VARS = (
    "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "AI_PROVIDER",
    "ADZUNA_APP_ID", "ADZUNA_APP_KEY", "RAPIDAPI_KEY", "SERPAPI_KEY",
    "GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET",
    "MICROSOFT_OAUTH_CLIENT_ID", "MICROSOFT_OAUTH_CLIENT_SECRET",
    "EMAIL_OAUTH_REDIRECT_BASE",
    "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM",
    "NOTIFY_EMAIL_TO", "NOTIFY_SEND_EMPTY",
)


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch):
    """Tests must never use (or spend quota on) the developer's real .env keys."""
    monkeypatch.setattr("app.config.load_dotenv", lambda *args, **kwargs: False)
    for var in API_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
