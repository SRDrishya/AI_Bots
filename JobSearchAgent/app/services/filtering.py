"""Post-fetch refinement: title cleanup -> location filter -> de-duplication.

Location rule (agreed requirement): keep jobs in the target city (Chennai)
plus remote jobs that someone living in India can actually take.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.config import CITY_ALIASES
from app.models import JobLink, JobListing

# --------------------------------------------------------------------------- #
# Title cleanup
# --------------------------------------------------------------------------- #
_TITLE_NOISE = re.compile(
    r"\b(hiring alert|urgent(ly)? hiring|urgent requirement|urgent opening|we are hiring|"
    r"hiring now|now hiring|immediate joiners?|walk[- ]?in drive|apply now)\b",
    re.IGNORECASE,
)


def clean_title(title: str) -> str:
    t = _TITLE_NOISE.sub(" ", title)
    t = re.sub(r"([!?*#|])\1+", " ", t)        # "!!", "**", "||"
    t = re.sub(r"\s*!+\s*", " ", t)            # stray "!"
    t = re.sub(r"\(\s*\)|\[\s*\]", " ", t)     # empty brackets left behind
    t = re.sub(r"\s{2,}", " ", t).strip(" -|:,;")
    return t or title.strip()


# --------------------------------------------------------------------------- #
# Location classification
# --------------------------------------------------------------------------- #
_INDIA_PLACES = [
    "india", "hyderabad", "pune", "delhi", "new delhi", "ncr", "noida", "kolkata", "coimbatore",
    "kochi", "cochin", "ahmedabad", "trivandrum", "thiruvananthapuram", "mysore", "mysuru",
    "jaipur", "chandigarh", "indore", "bhubaneswar", "visakhapatnam", "vizag", "madurai",
    "trichy", "tiruchirappalli", "tamil nadu", "karnataka", "telangana", "maharashtra", "kerala",
    *(alias for aliases in CITY_ALIASES.values() for alias in aliases),
]
# Remote-region words meaning "you can do this from India".
_INDIA_COMPATIBLE = re.compile(
    r"\b(india|anywhere|worldwide|global|globally|apac|asia|asia[- ]pacific|south asia|ist|"
    r"(utc|gmt) ?\+ ?0?5:?30)\b",
    re.IGNORECASE,
)
# Words that say nothing about *where* a remote job can be done.
_REMOTE_FILLER = re.compile(
    r"\b(remote|remote[- ]friendly|remote[- ]first|fully|100|work from home|wfh|flexible|"
    r"hybrid|travel[- ]required|optional|only|based|position|opportunity|n/?a|not specified)\b",
    re.IGNORECASE,
)
# Remote job descriptions that quietly restrict where you must live / be authorised.
_US_EU_ONLY = re.compile(
    r"(?:\b(?:US|U\.S\.|USA)\b|(?i:\b(?:united states|canada|uk|united kingdom|europe|european union|eu)\b))"
    r"[- ](?i:based|only|residents?|citizens?)\b"
    r"|(?i:\b(?:must|need to|required to)\b(?: be)? (?:reside|live|located|based)\b[^.\n]{0,40}?)"
    r"(?:\b(?:US|U\.S\.|USA)\b|(?i:\b(?:united states|canada|uk|united kingdom|europe|eu)\b))"
    r"|(?i:authori[sz]ed to work in the )(?:US|U\.S\.|(?i:united states|uk|united kingdom|canada))\b"
)
_UNKNOWN_LOCATIONS = {"", "not specified", "n/a", "na", "hybrid", "on-site", "onsite", "in office",
                      "in-office", "multiple locations", "various", "tbd", "remote"}


def _alias_regex(aliases: list[str]) -> re.Pattern[str]:
    return re.compile(r"\b(" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + r")\b",
                      re.IGNORECASE)


_INDIA_RE = _alias_regex(_INDIA_PLACES)


@dataclass
class LocationRule:
    city: str            # "" means no city preference (remote search)
    remote_only: bool = False

    def __post_init__(self):
        key = self.city.strip().lower()
        aliases = next((a for a in CITY_ALIASES.values() if key in a), [key] if key else [])
        self.city_re = _alias_regex(aliases) if aliases else None


def classify(job: JobListing, rule: LocationRule) -> tuple[str, str]:
    """Return (match, reason). match is "local" | "remote" | "" (excluded, see reason)."""
    loc = job.location.strip()
    loc_l = loc.lower()
    is_local = bool(rule.city_re and (rule.city_re.search(loc) or rule.city_re.search(job.title)))
    is_remote = job.remote or bool(re.search(r"\bremote\b|work from home|\bwfh\b", loc_l))

    if is_local and not rule.remote_only:
        return "local", ""

    if is_remote:
        if _INDIA_COMPATIBLE.search(loc) or (rule.city_re and rule.city_re.search(loc)):
            return "remote", ""
        residual = re.sub(r"[^a-z]+", " ", _REMOTE_FILLER.sub(" ", loc_l)).strip()
        if residual:  # e.g. "Remote (USA)", "Remote (Berlin)", "Remote-Friendly, United States"
            return "", "remote_restricted"
        if _US_EU_ONLY.search(job.description[:4000]):
            return "", "remote_restricted"
        return "remote", ""  # plain "Remote"

    if is_local:  # local but remote_only requested
        return "", "not_remote"
    if loc_l in _UNKNOWN_LOCATIONS:
        return "", "unknown_location"
    if _INDIA_RE.search(loc):
        return "", "other_city"
    return "", "other_country"


# --------------------------------------------------------------------------- #
# De-duplication
# --------------------------------------------------------------------------- #
_COMPANY_SUFFIXES = {
    "pvt", "private", "ltd", "limited", "llp", "llc", "inc", "corp", "corporation", "co", "company",
    "india", "technologies", "technology", "tech", "solutions", "services", "systems", "software",
    "global", "group", "holdings", "the", "plc", "gmbh", "consultancy", "consulting", "labs",
}
_TITLE_SYNONYMS = {"sr": "senior", "jr": "junior", "engg": "engineer", "enginner": "engineer",
                   "developer": "engineer", "dev": "engineer", "mgr": "manager"}
_TITLE_DROP = {"remote", "hybrid", "onsite", "full", "time", "fulltime", "contract", "permanent",
               "wfh", "work", "from", "home", "the", "a", "an", "for", "in", "at", "with"}
_ATS = {"greenhouse", "lever", "ashby"}
_SOURCE_RANK = {**dict.fromkeys(_ATS, 3), "serpapi": 2, "jsearch": 2, "adzuna": 2}
TITLE_SIMILARITY = 0.88


def normalize_company(name: str) -> str:
    words = re.sub(r"[^a-z0-9 ]+", " ", name.lower()).split()
    while words and words[0].isdigit():  # "5100 Kyndryl Solutions" (Workday entity codes)
        words.pop(0)
    original = " ".join(words)
    while len(words) > 1 and words[-1] in _COMPANY_SUFFIXES:
        words.pop()
    if len(words) == 1 and words[0] in _COMPANY_SUFFIXES:  # "The Company" -> keep as-is
        return original
    return " ".join(words) or name.lower().strip()


def normalize_title(title: str, city_re: re.Pattern[str] | None = None, company: str = "") -> str:
    t = clean_title(title).lower()
    if city_re:
        t = city_re.sub(" ", t)
    t = _INDIA_RE.sub(" ", t)
    company_words = set(company.split())  # "EPAM Sr Snowflake Engineer" posted by EPAM
    words = [_TITLE_SYNONYMS.get(w, w) for w in re.findall(r"[a-z0-9+#]+", t)]
    return " ".join(w for w in words if w not in _TITLE_DROP and w not in company_words)


def _rank(job: JobListing) -> tuple:
    return (job.location_match == "local", _SOURCE_RANK.get(job.source, 1),
            bool(job.salary), len(job.description))


def _merge(cluster: list[JobListing]) -> JobListing:
    best = max(cluster, key=_rank)
    merged = best.model_copy(deep=True)
    seen_urls = {merged.url}
    for other in cluster:
        if other is best:
            continue
        if other.url and other.url not in seen_urls:
            seen_urls.add(other.url)
            merged.alt_links.append(JobLink(source=other.source, url=other.url))
        for f in ("salary", "posted_at", "employment_type"):
            if not getattr(merged, f) and getattr(other, f):
                setattr(merged, f, getattr(other, f))
        if len(other.description) > len(merged.description) * 1.5:
            merged.description = other.description
        merged.tags = list(dict.fromkeys([*merged.tags, *other.tags]))
    return merged


def deduplicate(jobs: list[JobListing], city_re: re.Pattern[str] | None = None) -> tuple[list[JobListing], int]:
    """Cluster jobs that are the same opening (same company + same/similar title,
    or same URL) and merge each cluster into its best record."""
    clusters: list[list[JobListing]] = []
    by_company: dict[str, list[tuple[str, int]]] = {}
    by_url: dict[str, int] = {}

    for job in jobs:
        company = normalize_company(job.company)
        title = normalize_title(job.title, city_re, company)
        idx = by_url.get(job.url) if job.url else None
        if idx is None:
            for seen_title, cluster_idx in by_company.get(company, []):
                if seen_title == title or SequenceMatcher(None, seen_title, title).ratio() >= TITLE_SIMILARITY:
                    idx = cluster_idx
                    break
        if idx is None:
            idx = len(clusters)
            clusters.append([])
        clusters[idx].append(job)
        by_company.setdefault(company, []).append((title, idx))
        if job.url:
            by_url.setdefault(job.url, idx)

    merged = [_merge(c) if len(c) > 1 else c[0] for c in clusters]
    return merged, len(jobs) - len(merged)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class RefineResult:
    jobs: list[JobListing]
    excluded: Counter = field(default_factory=Counter)
    duplicates_merged: int = 0


def refine_jobs(jobs: list[JobListing], city: str, remote_only: bool = False) -> RefineResult:
    rule = LocationRule(city=city, remote_only=remote_only)
    kept: list[JobListing] = []
    excluded: Counter = Counter()
    for job in jobs:
        job = job.model_copy(update={"title": clean_title(job.title)})
        match, reason = classify(job, rule)
        if match:
            job.location_match = match
            kept.append(job)
        else:
            excluded[reason] += 1
    unique, merged_count = deduplicate(kept, rule.city_re)
    return RefineResult(jobs=unique, excluded=excluded, duplicates_merged=merged_count)
