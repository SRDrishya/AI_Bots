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
    adzuna_app_id: str
    adzuna_app_key: str
    rapidapi_key: str
    serpapi_key: str
    host: str
    port: int
    data_dir: Path


def get_env_settings() -> EnvSettings:
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
    return EnvSettings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL,
        gemini_fallback_models=[
            m.strip()
            for m in os.getenv("GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_FALLBACKS).split(",")
            if m.strip()
        ],
        adzuna_app_id=os.getenv("ADZUNA_APP_ID", "").strip(),
        adzuna_app_key=os.getenv("ADZUNA_APP_KEY", "").strip(),
        rapidapi_key=os.getenv("RAPIDAPI_KEY", "").strip(),
        serpapi_key=os.getenv("SERPAPI_KEY", "").strip(),
        host=os.getenv("HOST", "127.0.0.1").strip() or "127.0.0.1",
        port=int(os.getenv("PORT", "8000").strip() or "8000"),
        data_dir=Path(os.getenv("DATA_DIR", "").strip() or BASE_DIR / "data"),
    )
