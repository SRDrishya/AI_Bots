"""Job ↔ resume match scoring.

Hybrid strategy (agreed requirement):
  1. Local score for every job (fast, free, deterministic):
        40% title fit   — job title vs target roles / queries
        35% skill fit   — share of the job's skills the candidate has
        25% text fit    — TF-IDF cosine, resume vs description (relative to best job)
     × level fit        — penalty for seniority / required-years mismatch
     + small bonuses    — local (Chennai) role, posted in the last week
  2. Gemini re-scores the top N (default 25) in batches of 10, adding reasoning
     and matched/missing skills. Final = 75% Gemini + 25% local. Any Gemini
     failure leaves the local scores in place.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from app.models import JobListing, ResumeProfile, ScoredJob
from app.services.ai import ai_available, generate_json
from app.services.gemini import GeminiError
from app.services.skills import extract_skills, normalize_skill
from app.sources.base import _GENERIC, _words, clean_query  # shared title tokenisation

log = logging.getLogger(__name__)

W_TITLE, W_SKILLS, W_TEXT = 0.40, 0.35, 0.25
LOCAL_BONUS, RECENT_BONUS = 3, 2
GEMINI_TOP_N = 25
GEMINI_BATCH = 10
GEMINI_WEIGHT = 0.75
GEMINI_CONCURRENCY = 3
GEMINI_DESC_CHARS = 1800

# --------------------------------------------------------------------------- #
# Title fit
# --------------------------------------------------------------------------- #
_SOFTWARE_NOUNS = {"engineer", "architect"}


def title_fit(title: str, roles: list[str]) -> tuple[float, str]:
    """Best fraction of a role's core words present in the title (0..1), and that role."""
    words = set(_words(title))
    best, best_role = 0.0, ""
    for role in roles:
        core = {w for w in _words(role) if w not in _GENERIC}
        if not core:
            continue
        frac = len(core & words) / len(core)
        if core == {"software", "engineer"} and "engineer" in words:
            frac = max(frac, 0.75)  # any engineering title is a decent fit for a generalist
        if frac > best:
            best, best_role = frac, role
    return best, best_role


# --------------------------------------------------------------------------- #
# Seniority / experience fit
# --------------------------------------------------------------------------- #
_LEVELS = [
    (0, re.compile(r"\b(intern|internship|trainee|fresher|graduate|apprentice)\b", re.I)),
    (4, re.compile(r"\b(lead|staff|principal|architect|head|director|manager|vp)\b", re.I)),
    (3, re.compile(r"\b(senior|sr\.?|iii|iv)\b", re.I)),
    (1, re.compile(r"\b(junior|jr\.?|associate|entry)\b", re.I)),
]
_PROFILE_LEVEL = {"entry": 0, "junior": 1, "mid": 2, "senior": 3, "lead": 4}
_YEARS_REQUIRED = re.compile(
    r"(?<![\d.])(\d{1,2})\s*\+?(?:\s*(?:-|–|to)\s*\d{1,2})?\s*\+?\s*(?:years?|yrs?)\b",
    re.I,
)


def job_level(title: str) -> int | None:
    for level, pattern in _LEVELS:
        if pattern.search(title):
            return level
    return None  # unspecified


def candidate_level(profile: ResumeProfile) -> int | None:
    if profile.seniority.lower() in _PROFILE_LEVEL:
        return _PROFILE_LEVEL[profile.seniority.lower()]
    y = profile.years_of_experience
    if y is None:
        return None
    return 0 if y < 1 else 1 if y < 3 else 2 if y < 6 else 3 if y < 10 else 4


def years_required(description: str) -> int | None:
    """Smallest plausible 'N years' requirement in the first part of the description."""
    found = [int(m.group(1)) for m in _YEARS_REQUIRED.finditer(description[:5000])]
    found = [y for y in found if 0 < y <= 20]
    return min(found) if found else None


def level_fit(job: JobListing, profile: ResumeProfile) -> tuple[float, list[str]]:
    notes: list[str] = []
    factor = 1.0
    cand = candidate_level(profile)
    lvl = job_level(job.title)
    if cand is not None and lvl is not None:
        gap = lvl - cand
        if gap >= 2:
            factor *= 0.55
            notes.append("role is well above your level")
        elif gap == 1:
            factor *= 0.85
        elif gap <= -2:
            factor *= 0.7
            notes.append("role is below your level")
    req = years_required(job.description)
    yrs = profile.years_of_experience
    if req is not None and yrs is not None:
        if req > yrs + 1:
            factor *= max(0.5, 1 - 0.08 * (req - yrs))
            notes.append(f"asks {req}+ yrs (you: {yrs:g})")
        elif req > yrs:
            notes.append(f"asks {req}+ yrs (you: {yrs:g}, close)")
        else:
            notes.append(f"asks {req}+ yrs ✓")
    return factor, notes


# --------------------------------------------------------------------------- #
# TF-IDF (tiny, dependency-free)
# --------------------------------------------------------------------------- #
_STOP = set("""a an and are as at be been but by can for from has have in into is it its of on or our
our that the their them they this to was we will with you your about across all also any who what
when where which while within work working team teams role roles job jobs company experience years
strong ability skills knowledge including using use well new help join etc must should would able""".split())


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z0-9+#.]*[a-z0-9+#]|[a-z]", text.lower())
            if t not in _STOP and len(t) > 1]


def tfidf_similarities(query: str, docs: list[str]) -> list[float]:
    tokenised = [Counter(_tokens(d)) for d in docs]
    q = Counter(_tokens(query))
    n = len(docs) + 1
    df = Counter()
    for tf in [*tokenised, q]:
        df.update(tf.keys())
    idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}

    def vec(tf: Counter) -> dict[str, float]:
        v = {t: (1 + math.log(c)) * idf[t] for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    qv = vec(q)
    return [sum(qv.get(t, 0.0) * x for t, x in vec(tf).items()) for tf in tokenised]


# --------------------------------------------------------------------------- #
# Local scoring
# --------------------------------------------------------------------------- #
def _profile_document(profile: ResumeProfile) -> str:
    return " ".join([
        " ".join(profile.target_roles) * 2,
        " ".join(profile.skills) * 2,
        profile.summary,
        profile.raw_text_preview,
    ])


def _is_recent(posted_at: str, days: int = 7) -> bool:
    try:
        dt = datetime.fromisoformat(posted_at)
    except ValueError:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - dt <= timedelta(days=days)


def _sentence(text: str) -> str:
    return (text[:1].upper() + text[1:] + ".") if text else ""


def score_local(jobs: list[JobListing], profile: ResumeProfile) -> list[ScoredJob]:
    if not jobs:
        return []
    roles = list(dict.fromkeys([*profile.target_roles, *map(clean_query, profile.search_queries)]))
    have = {s.lower() for s in profile.skills}

    sims = tfidf_similarities(_profile_document(profile), [f"{j.title} {j.title} {j.description}" for j in jobs])
    top_sim = max(sims) or 1.0

    scored: list[ScoredJob] = []
    for job, sim in zip(jobs, sims):
        t_fit, role = title_fit(job.title, roles)

        job_skills = extract_skills(f"{job.title}\n{job.description}\n{' '.join(job.tags)}")
        matched = [s for s in job_skills if s.lower() in have]
        missing = [s for s in job_skills if s.lower() not in have][:8]
        if job_skills:
            s_fit = min(1.0, len(matched) / min(len(job_skills), 8))
        else:
            s_fit = 0.5  # no detectable skills (very short description) -> neutral

        x_fit = sim / top_sim
        lvl, notes = level_fit(job, profile)

        raw = (W_TITLE * t_fit + W_SKILLS * s_fit + W_TEXT * x_fit) * 100 * lvl
        if job.location_match == "local":
            raw += LOCAL_BONUS
        if _is_recent(job.posted_at):
            raw += RECENT_BONUS
        score = max(0, min(100, round(raw)))

        reasons = []
        if role and t_fit >= 0.99:
            reasons.append(f"title matches '{role}'")
        elif role and t_fit > 0:
            reasons.append(f"title partly matches '{role}'")
        else:
            reasons.append("title differs from your target roles")
        if job_skills:
            reasons.append(f"{len(matched)}/{len(job_skills)} listed skills matched")
        reasons.extend(notes)

        scored.append(ScoredJob(
            **job.model_dump(),
            match_score=score,
            matched_skills=matched,
            missing_skills=missing,
            match_reasoning=_sentence("; ".join(reasons)),
            scoring_engine="local",
        ))
    scored.sort(key=lambda j: j.match_score, reverse=True)
    return scored


# --------------------------------------------------------------------------- #
# Gemini re-ranking
# --------------------------------------------------------------------------- #
class _GeminiJobScore(BaseModel):
    id: str
    score: int = Field(description="0-100 fit of this job for this candidate")
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list, description="Important requirements the candidate lacks")
    reasoning: str = Field(default="", description="1-2 sentences, specific to this job and candidate")


class _GeminiBatch(BaseModel):
    results: list[_GeminiJobScore]


_RANK_PROMPT = """You are an experienced technical recruiter in India. Score how well each job fits this candidate.

CANDIDATE
Target roles: {roles}
Seniority: {seniority}; experience: {years} years
Skills: {skills}
Summary: {summary}
Location preference: {locations}

SCORING GUIDE (0-100)
90+  excellent fit: role, level and core skills all match
75-89 strong fit: minor gaps
60-74 reasonable: some important gaps or level mismatch
40-59 weak: significant gaps
<40  poor fit
Weigh: role/function match > core required skills > seniority & years > nice-to-haves.
Penalise jobs whose required experience clearly exceeds the candidate's.
Return one result per job, using the job's id exactly. missing_skills: at most 5, only real requirements.

JOBS
{jobs}"""


def _job_block(key: str, job: ScoredJob) -> str:
    desc = re.sub(r"\s+", " ", job.description)[:GEMINI_DESC_CHARS]
    return (f"--- id: {key}\nTitle: {job.title}\nCompany: {job.company}\nLocation: {job.location}\n"
            f"Description: {desc}")


async def _rank_batch(batch: list[ScoredJob], profile: ResumeProfile, api_key: str, model: str) -> dict[str, _GeminiJobScore]:
    keys = {f"j{i + 1}": job for i, job in enumerate(batch)}
    prompt = _RANK_PROMPT.format(
        roles=", ".join(profile.target_roles) or "not specified",
        seniority=profile.seniority or "unknown",
        years=f"{profile.years_of_experience:g}" if profile.years_of_experience is not None else "unknown",
        skills=", ".join(profile.skills[:30]) or "not specified",
        summary=profile.summary[:600] or "n/a",
        locations=", ".join(profile.preferred_locations) or "Chennai, Remote",
        jobs="\n".join(_job_block(k, j) for k, j in keys.items()),
    )
    result = await generate_json(api_key=api_key, model=model, prompt=prompt, schema=_GeminiBatch, temperature=0.1,
                                 fast=True)
    return {keys[r.id].id: r for r in result.results if r.id in keys}


def _apply_gemini(job: ScoredJob, g: _GeminiJobScore) -> ScoredJob:
    ai_score = max(0, min(100, g.score))
    return job.model_copy(update={
        "match_score": round(GEMINI_WEIGHT * ai_score + (1 - GEMINI_WEIGHT) * job.match_score),
        "matched_skills": [normalize_skill(s) for s in g.matched_skills if s.strip()] or job.matched_skills,
        "missing_skills": [normalize_skill(s) for s in g.missing_skills if s.strip()][:5] or job.missing_skills,
        "match_reasoning": g.reasoning.strip() or job.match_reasoning,
        "scoring_engine": "gemini",
    })


@dataclass
class ScoringResult:
    jobs: list[ScoredJob]
    engine: str  # "local" | "gemini" | "local (gemini failed: ...)" | "gemini (partial: ...)"


async def score_jobs(
    jobs: list[JobListing],
    profile: ResumeProfile,
    gemini_api_key: str = "",
    gemini_model: str = "",
    top_n: int = GEMINI_TOP_N,
) -> ScoringResult:
    local = score_local(jobs, profile)
    if not ai_available(gemini_api_key) or not local or top_n <= 0:
        return ScoringResult(local, "local")

    head = local[:top_n]
    batches = [head[i:i + GEMINI_BATCH] for i in range(0, len(head), GEMINI_BATCH)]
    semaphore = asyncio.Semaphore(GEMINI_CONCURRENCY)

    async def run(batch: list[ScoredJob]):
        async with semaphore:
            return await _rank_batch(batch, profile, gemini_api_key, gemini_model)

    outcomes = await asyncio.gather(*(run(b) for b in batches), return_exceptions=True)
    ai: dict[str, _GeminiJobScore] = {}
    errors: list[str] = []
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            log.warning("Gemini scoring batch failed: %s", outcome)
            errors.append(str(outcome) if isinstance(outcome, GeminiError) else type(outcome).__name__)
        else:
            ai.update(outcome)

    if not ai:
        return ScoringResult(local, f"local (gemini failed: {errors[0] if errors else 'no results'})")

    final = [_apply_gemini(j, ai[j.id]) if j.id in ai else j for j in local]
    final.sort(key=lambda j: (j.match_score, j.scoring_engine == "gemini"), reverse=True)
    engine = "gemini" if not errors else f"gemini (partial: {errors[0]})"
    return ScoringResult(final, engine)
