"""Shared plumbing for job source adapters."""
from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from app.config import CITY_ALIASES, get_env_settings
from app.models import ApiKeysInput, ATSBoardConfig, JobListing, ResumeProfile
from app.sources.quota import QuotaTracker

USER_AGENT = "JobSearchAgent/0.1 (personal job search tool)"
MAX_DESCRIPTION_CHARS = 8000


class SourceSkipped(Exception):
    """Raised by a source that can't run (e.g. missing API key). Not an error."""


# --------------------------------------------------------------------------- #
# Search context
# --------------------------------------------------------------------------- #
def resolve_keys(overrides: ApiKeysInput | None) -> ApiKeysInput:
    """Per-request keys win; anything blank falls back to .env."""
    env = get_env_settings()
    o = overrides or ApiKeysInput()
    return ApiKeysInput(
        gemini_api_key=o.gemini_api_key.strip() or env.gemini_api_key,
        gemini_model=o.gemini_model.strip() or env.gemini_model,
        adzuna_app_id=o.adzuna_app_id.strip() or env.adzuna_app_id,
        adzuna_app_key=o.adzuna_app_key.strip() or env.adzuna_app_key,
        rapidapi_key=o.rapidapi_key.strip() or env.rapidapi_key,
        serpapi_key=o.serpapi_key.strip() or env.serpapi_key,
    )


@dataclass
class SearchContext:
    queries: list[str]
    location: str
    remote_only: bool
    keys: ApiKeysInput
    ats_boards: list[ATSBoardConfig]
    per_source_limit: int
    matcher: "TitleMatcher" = field(repr=False, default=None)  # type: ignore[assignment]
    quota: "QuotaTracker | None" = field(repr=False, default=None)  # None = no quota guard

    @property
    def city(self) -> str:
        """First component of the location, e.g. 'Chennai' from 'Chennai, India'."""
        loc = self.location.split(",")[0].strip()
        return "" if loc.lower().startswith("remote") else loc


_QUERY_NOISE = re.compile(
    r"\b(remote|hybrid|on-?site|wfh|work from home|jobs?|openings?|vacanc(?:y|ies)|hiring|"
    r"in|near|at|india|tamil nadu|" + "|".join(re.escape(a) for v in CITY_ALIASES.values() for a in v) + r")\b",
    re.IGNORECASE,
)


def clean_query(query: str) -> str:
    """Strip locations / 'remote' / 'jobs' — sources add location themselves, and
    the title matcher would otherwise require e.g. 'chennai' in job titles."""
    cleaned = re.sub(r"\s{2,}", " ", _QUERY_NOISE.sub(" ", query)).strip(" ,-|/")
    return cleaned or query.strip()


def build_queries(profile: ResumeProfile, custom_query: str) -> list[str]:
    if custom_query.strip():
        raw = custom_query.split(";")
    else:
        raw = profile.search_queries or profile.target_roles or [f"{s} developer" for s in profile.skills[:3]]
    return list(dict.fromkeys(clean_query(q) for q in raw if q.strip()))


# --------------------------------------------------------------------------- #
# Relevance filter for sources that return everything (ATS boards, Arbeitnow)
# or whose search is loose (Remotive).
# --------------------------------------------------------------------------- #
# Seniority / filler words don't change what the job *is*; role nouns do
# ("Data Engineer" must not match "Data Scientist").
_GENERIC = {
    "senior", "sr", "sr.", "junior", "jr", "jr.", "lead", "principal", "staff", "associate",
    "intern", "i", "ii", "iii", "iv", "and", "of", "the", "for", "in", "with", "a",
}
_SYNONYMS = {"developer": "engineer", "programmer": "engineer", "dev": "engineer", "sde": "engineer",
             "backend": "back-end", "frontend": "front-end", "fullstack": "full-stack"}
_SOFTWARE_TITLE = re.compile(r"\b(engineer|developer|programmer|sde)\b", re.IGNORECASE)
_BROAD_SOFTWARE = {"software", "engineer"}


def _words(text: str) -> list[str]:
    lowered = re.sub(r"\b(back|front|full) (end|stack)\b", r"\1-\2", text.lower())
    return [_SYNONYMS.get(w, w) for w in re.findall(r"[a-z0-9+#.\-]+", lowered)]


class TitleMatcher:
    """A title is relevant if it contains every core word of some query/role,
    or names one of the candidate's top skills."""

    def __init__(self, queries: list[str], roles: list[str], skills: list[str]):
        self.groups: list[set[str]] = []
        self.broad_software = False
        for phrase in [*queries, *roles]:
            core = {w for w in _words(phrase) if w not in _GENERIC}
            if core <= _BROAD_SOFTWARE and _SOFTWARE_TITLE.search(phrase):
                self.broad_software = True  # "Software Engineer" -> any engineer/developer title
            elif core:
                self.groups.append(core)
        self.skills = {s.lower() for s in skills[:8] if len(s) > 2}

    def matches(self, title: str) -> bool:
        words = set(_words(title))
        if any(group <= words for group in self.groups):
            return True
        lowered = title.lower()
        for skill in self.skills:
            if (" " in skill and skill in lowered) or skill in words:
                return True
        return self.broad_software and bool(_SOFTWARE_TITLE.search(title))


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
_TAG = re.compile(r"<[^>]+>")
_BLOCK = re.compile(r"</?(p|div|br|li|ul|ol|h[1-6]|tr)[^>]*>", re.IGNORECASE)
_REMOTE = re.compile(r"\b(remote|work from home|wfh|anywhere|distributed)\b", re.IGNORECASE)


def html_to_text(raw: str | None) -> str:
    if not raw:
        return ""
    text = html.unescape(raw)
    if "&lt;" in raw and "<" not in raw:  # Greenhouse double-escapes
        text = html.unescape(text)
    text = _BLOCK.sub("\n", text)
    text = html.unescape(_TAG.sub(" ", text))
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()[:MAX_DESCRIPTION_CHARS]


def looks_remote(*texts: str | None) -> bool:
    return any(t and _REMOTE.search(t) for t in texts)


_RELATIVE = re.compile(r"(\d+)\+?\s*(minute|min|hour|hr|day|week|month)s?\s+ago", re.IGNORECASE)
_UNIT_SECONDS = {"minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400, "week": 604800, "month": 2592000}


def iso_date(value: str | int | float | None) -> str:
    """Normalise unix seconds/ms, ISO-ish strings, or Google-style relative
    dates ("12 days ago") to ISO 8601 (UTC)."""
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if m := _RELATIVE.search(lowered):
            seconds = int(m.group(1)) * _UNIT_SECONDS[m.group(2).lower()]
            return datetime.fromtimestamp(time.time() - seconds, tz=timezone.utc).isoformat(timespec="seconds")
        if lowered in {"today", "just posted", "just now"}:
            return datetime.now(timezone.utc).isoformat(timespec="seconds")
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        ts = float(value)
        if ts > 1e12:  # milliseconds
            ts /= 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)  # e.g. "3 days ago" from Google Jobs — keep as-is
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def format_salary(lo: float | None, hi: float | None, currency: str | None = "", period: str | None = "") -> str:
    if not lo and not hi:
        return ""

    def fmt(v: float) -> str:
        if v >= 100_000 and (currency or "").upper() == "INR":
            return f"{v / 100_000:.1f}L"  # lakhs
        if v >= 1000:
            return f"{v / 1000:.0f}K"
        return f"{v:g}"

    amount = " - ".join(fmt(v) for v in (lo, hi) if v)
    if lo and hi and lo == hi:
        amount = fmt(lo)
    suffix = f" / {period}" if period else ""
    return f"{(currency or '').upper()} {amount}{suffix}".strip()


def make_job(source: str, raw_id: str | int, **fields) -> JobListing:
    fields["title"] = (fields.get("title") or "").strip() or "Untitled role"
    fields["company"] = (fields.get("company") or "").strip() or "Unknown company"
    fields["location"] = (fields.get("location") or "").strip() or ("Remote" if fields.get("remote") else "Not specified")
    fields["tags"] = [t for t in dict.fromkeys(str(t).strip() for t in fields.get("tags") or []) if t]
    return JobListing(id=f"{source}:{raw_id}", source=source, **fields)


_CACHE: dict[str, tuple[float, object]] = {}


def clear_cache() -> None:
    _CACHE.clear()


async def get_json(client: httpx.AsyncClient, url: str, *, cache_ttl: float = 0, **kwargs):
    """GET + JSON decode. With cache_ttl > 0, identical requests are served from memory
    (used for free APIs that ask clients not to poll frequently)."""
    key = ""
    if cache_ttl:
        params = sorted((kwargs.get("params") or {}).items())
        key = f"{url}?{params}"
        hit = _CACHE.get(key)
        if hit and hit[0] > time.monotonic():
            return hit[1]
    response = await client.get(url, **kwargs)
    response.raise_for_status()
    data = response.json()
    if cache_ttl:
        _CACHE[key] = (time.monotonic() + cache_ttl, data)
    return data


def new_client(timeout: float = 20.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=10.0),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        follow_redirects=True,
        limits=httpx.Limits(max_connections=20),
    )
