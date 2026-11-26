from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

# Same model setup as JobSearchAgent: Gemini first, Claude as fallback.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GEMINI_FALLBACKS = "gemini-3.5-flash-lite"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"        # answers, web search
DEFAULT_ANTHROPIC_FAST_MODEL = "claude-haiku-5-5"  # extraction, duplicate checks, scoring practice answers
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-2"
AI_PROVIDERS = ("gemini", "anthropic")

DEFAULT_QUESTIONS_PER_TECH = 100
ANSWER_BATCH_SIZE = 4      # questions answered per AI call (detailed answers are long)
AI_CONCURRENCY = 3         # parallel AI calls per build job
PAGE_CONCURRENCY = 5       # parallel page downloads
MAX_PAGES_PER_ROUND = 15   # question pages read per search round (direct report pages + search results)
MAX_CHUNKS_PER_PAGE = 4    # ~12k characters each; long pages are cut after this
MAX_SEARCH_ROUNDS = 4      # a full build searches again (different queries) until it reaches the target

# Companies whose candidates' interview reports are searched first (editable in Settings).
DEFAULT_TARGET_COMPANIES = [
    # IT services
    "TCS", "Infosys", "Cognizant", "Accenture", "Wipro", "HCLTech", "Capgemini", "LTIMindtree", "Virtusa",
    # Product companies
    "Zoho", "Freshworks", "Chargebee", "PayPal", "Microsoft", "Amazon",
    # Banking / insurance tech
    "Citi", "Barclays", "Wells Fargo", "Standard Chartered", "HSBC",
]

# The resume saved by JobSearchAgent is imported on first start if we have none yet.
JOBSEARCH_RESUME_FILE = BASE_DIR.parent / "JobSearchAgent" / "data" / "resume_text.json"


@dataclass(frozen=True)
class Experience:
    key: str           # used in file names and URLs
    label: str         # shown in the UI
    phrase: str        # used in search queries ("5 years experience")
    expectation: str   # what interviewers expect at this level (steers answer depth)


EXPERIENCE_LEVELS: list[Experience] = [
    Experience("0-2", "0–2 years (Junior)", "2 years experience",
               "solid fundamentals, clear definitions, simple correct examples"),
    Experience("3-4", "3–4 years (Mid-level)", "3 years experience",
               "fundamentals plus practical usage, common pitfalls, debugging and real project examples"),
    Experience("5-7", "5–7 years (Senior)", "5 years experience",
               "internals and the 'why', trade-offs, performance, production issues, design decisions, "
               "mentoring others and leading features"),
    Experience("8-10", "8–10 years (Lead)", "8 years experience",
               "architecture and system-level trade-offs, scalability, team practices, technical leadership "
               "and decision making across services"),
    Experience("11-plus", "11+ years (Architect)", "10+ years experience",
               "enterprise architecture, strategy, cross-team standards, cost/risk trade-offs, leading large "
               "migrations"),
]
DEFAULT_EXPERIENCE = "5-7"


def get_experience(key: str) -> Experience | None:
    return next((e for e in EXPERIENCE_LEVELS if e.key == key), None)


@dataclass(frozen=True)
class EnvSettings:
    gemini_api_key: str
    gemini_model: str
    gemini_fallback_models: list[str]
    anthropic_api_key: str
    anthropic_model: str
    anthropic_fast_model: str
    ai_provider: str  # main provider; the other one (if its key is set) is the fallback
    embedding_model: str
    claude_code_model: str   # empty = Claude Code's default model
    questions_per_tech: int
    host: str
    port: int
    data_dir: Path


def _ai_provider() -> str:
    value = os.getenv("AI_PROVIDER", "").strip().lower()
    value = "anthropic" if value == "claude" else value
    return value if value in AI_PROVIDERS else "gemini"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def get_env_settings() -> EnvSettings:
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
    return EnvSettings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL,
        gemini_fallback_models=[
            m.strip()
            for m in os.getenv("GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_FALLBACKS).split(",")
            if m.strip()
        ],
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "").strip() or DEFAULT_ANTHROPIC_MODEL,
        anthropic_fast_model=os.getenv("ANTHROPIC_FAST_MODEL", "").strip() or DEFAULT_ANTHROPIC_FAST_MODEL,
        ai_provider=_ai_provider(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL,
        claude_code_model=os.getenv("CLAUDE_CODE_MODEL", "").strip(),
        questions_per_tech=max(10, _int_env("QUESTIONS_PER_TECH", DEFAULT_QUESTIONS_PER_TECH)),
        host=os.getenv("HOST", "127.0.0.1").strip() or "127.0.0.1",
        port=_int_env("PORT", 8010),
        data_dir=Path(os.getenv("DATA_DIR", "").strip() or BASE_DIR / "data"),
    )
