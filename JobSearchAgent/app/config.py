from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


# gemini-2.5-* are no longer available to new API keys (404). Fallbacks are
# tried in order when the main model is overloaded (503) or unavailable.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GEMINI_FALLBACKS = "gemini-3.5-flash-lite"

# Anthropic Claude: alternative / fallback AI provider (ANTHROPIC_API_KEY).
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"        # writing: cover letters, tailored resumes, company research
DEFAULT_ANTHROPIC_FAST_MODEL = "claude-haiku-5-5"  # high-volume extraction: resume parsing, job scoring
AI_PROVIDERS = ("gemini", "anthropic")

# India has no daylight saving, so a fixed +05:30 offset is exact (no tz database needed).
IST = timezone(timedelta(hours=5, minutes=30), "IST")

# Target market: Chennai jobs (on-site/hybrid) plus remote roles anywhere.
DEFAULT_PREFERRED_LOCATIONS = ["Chennai", "Remote"]

# Names that mean "this job is in <city>" — old names, IT corridors, suburbs.
CITY_ALIASES: dict[str, list[str]] = {
    "chennai": [
        "chennai", "madras", "omr", "old mahabalipuram road", "guindy", "sholinganallur",
        "taramani", "siruseri", "perungudi", "thoraipakkam", "karapakkam", "navalur",
        "kelambakkam", "velachery", "porur", "ambattur", "tambaram", "chromepet",
        "pallavaram", "alandur", "ekkaduthangal", "nungambakkam", "teynampet", "egmore",
        "anna nagar", "t nagar", "t. nagar", "mylapore", "adyar", "mogappair", "manapakkam",
        "kandanchavadi", "perumbakkam", "medavakkam", "tidel park", "mahindra world city",
        "sriperumbudur", "oragadam", "guduvanchery",
    ],
    "bangalore": ["bangalore", "bengaluru"],
    "gurgaon": ["gurgaon", "gurugram"],
    "mumbai": ["mumbai", "bombay", "navi mumbai", "thane"],
}

MAX_RESUME_BYTES = 5 * 1024 * 1024  # 5 MB


STARTER_ATS_BOARDS = [
    {"provider": "greenhouse", "slug": "stripe", "label": "Stripe", "enabled": True},
    {"provider": "greenhouse", "slug": "anthropic", "label": "Anthropic", "enabled": True},
    {"provider": "greenhouse", "slug": "figma", "label": "Figma", "enabled": True},
    {"provider": "greenhouse", "slug": "databricks", "label": "Databricks", "enabled": True},
    {"provider": "greenhouse", "slug": "cloudflare", "label": "Cloudflare", "enabled": True},
    {"provider": "lever", "slug": "spotify", "label": "Spotify", "enabled": True},
    {"provider": "lever", "slug": "palantir", "label": "Palantir", "enabled": False},
    {"provider": "ashby", "slug": "openai", "label": "OpenAI", "enabled": True},
    {"provider": "ashby", "slug": "perplexity", "label": "Perplexity", "enabled": True},
    {"provider": "ashby", "slug": "linear", "label": "Linear", "enabled": True},
]


@dataclass(frozen=True)
class EnvSettings:
    gemini_api_key: str
    gemini_model: str
    gemini_fallback_models: list[str]
    anthropic_api_key: str
    anthropic_model: str
    anthropic_fast_model: str
    ai_provider: str  # main provider; the other one (if its key is set) is the fallback
    adzuna_app_id: str
    adzuna_app_key: str
    rapidapi_key: str
    serpapi_key: str
    google_oauth_client_id: str
    google_oauth_client_secret: str
    microsoft_oauth_client_id: str
    microsoft_oauth_client_secret: str
    email_oauth_redirect_base: str
    # New-jobs email after each search (SMTP). Off unless SMTP_HOST and NOTIFY_EMAIL_TO are set.
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    smtp_from: str
    notify_email_to: list[str]
    notify_send_empty: bool
    host: str
    port: int
    data_dir: Path


def _ai_provider() -> str:
    """AI_PROVIDER=gemini|anthropic (also accepts 'claude'). Default: Gemini first, Claude only as fallback.
    (With only a Claude key, the router skips Gemini, so Claude is still used.)"""
    value = os.getenv("AI_PROVIDER", "").strip().lower()
    value = "anthropic" if value == "claude" else value
    return value if value in AI_PROVIDERS else "gemini"


def get_env_settings() -> EnvSettings:
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
    port = int(os.getenv("PORT", "8000").strip() or "8000")
    email_redirect_base = os.getenv("EMAIL_OAUTH_REDIRECT_BASE", "").strip().rstrip("/")
    return EnvSettings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL,
        gemini_fallback_models=[
            m.strip()
            for m in os.getenv("GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_FALLBACKS).split(",")
            if m.strip()
        ],
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "").strip() or DEFAULT_ANTHROPIC_MODEL,
        anthropic_fast_model=os.getenv("ANTHROPIC_FAST_MODEL", "").strip() or DEFAULT_ANTHROPIC_FAST_MODEL,
        ai_provider=_ai_provider(),
        adzuna_app_id=os.getenv("ADZUNA_APP_ID", "").strip(),
        adzuna_app_key=os.getenv("ADZUNA_APP_KEY", "").strip(),
        rapidapi_key=os.getenv("RAPIDAPI_KEY", "").strip(),
        serpapi_key=os.getenv("SERPAPI_KEY", "").strip(),
        google_oauth_client_id=os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip(),
        google_oauth_client_secret=os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "").strip(),
        microsoft_oauth_client_id=os.getenv("MICROSOFT_OAUTH_CLIENT_ID", "").strip(),
        microsoft_oauth_client_secret=os.getenv("MICROSOFT_OAUTH_CLIENT_SECRET", "").strip(),
        email_oauth_redirect_base=email_redirect_base or f"http://127.0.0.1:{port}",
        smtp_host=os.getenv("SMTP_HOST", "").strip(),
        smtp_port=int(os.getenv("SMTP_PORT", "587").strip() or "587"),
        smtp_user=os.getenv("SMTP_USER", "").strip(),
        smtp_password=os.getenv("SMTP_PASSWORD", "").strip().replace(" ", ""),  # Gmail shows app passwords with spaces
        smtp_from=os.getenv("SMTP_FROM", "").strip(),
        notify_email_to=[a.strip() for a in os.getenv("NOTIFY_EMAIL_TO", "").split(",") if a.strip()],
        notify_send_empty=os.getenv("NOTIFY_SEND_EMPTY", "").strip().lower() in ("1", "true", "yes", "on"),
        host=os.getenv("HOST", "127.0.0.1").strip() or "127.0.0.1",
        port=port,
        data_dir=Path(os.getenv("DATA_DIR", "").strip() or BASE_DIR / "data"),
    )
