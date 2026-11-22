"""Key-based sources with real India / Chennai coverage. All optional.

Free tiers are small (JSearch ~200 req/month, SerpAPI ~250/month), so each
search makes at most 3 calls per source:
  * top 2 queries in the target city
  * top query as a remote search across India
With remote_only, every call is a remote/India call instead.
"""
from __future__ import annotations

import httpx

from app.models import JobListing
from app.sources.base import (
    SearchContext,
    SourceSkipped,
    format_salary,
    get_json,
    html_to_text,
    iso_date,
    looks_remote,
    make_job,
)

CITY_QUERIES = 2


def _plan(ctx: SearchContext) -> list[tuple[str, bool]]:
    """[(query, is_remote_search)] for this search."""
    queries = ctx.queries or []
    if not queries:
        return []
    if ctx.remote_only or not ctx.city:
        return [(q, True) for q in queries[:CITY_QUERIES]]
    return [(q, False) for q in queries[:CITY_QUERIES]] + [(queries[0], True)]


def _guarded_plan(ctx: SearchContext, source: str) -> list[tuple[str, bool]]:
    """The call plan, trimmed to the remaining free quota (skips the source if none left)."""
    plan = _plan(ctx)
    if ctx.quota is None:
        return plan
    allowed = ctx.quota.allow(source, len(plan))
    if allowed == 0:
        raise SourceSkipped(ctx.quota.skip_message(source))
    return plan[:allowed]


# --------------------------------------------------------------------------- #
# Adzuna — India endpoint (salaries in INR)
# --------------------------------------------------------------------------- #
async def fetch_adzuna(ctx: SearchContext, client: httpx.AsyncClient) -> list[JobListing]:
    if not (ctx.keys.adzuna_app_id and ctx.keys.adzuna_app_key):
        raise SourceSkipped("Set ADZUNA_APP_ID and ADZUNA_APP_KEY to enable")
    jobs: dict[str, JobListing] = {}
    for query, remote in _plan(ctx)[:CITY_QUERIES]:  # Adzuna's remote search is weak; skip it
        params = {
            "app_id": ctx.keys.adzuna_app_id,
            "app_key": ctx.keys.adzuna_app_key,
            "what": f"{query} remote" if remote else query,
            "results_per_page": 50,
            "max_days_old": 30,
            "content-type": "application/json",
        }
        if not remote:
            params["where"] = ctx.city
        data = await get_json(client, "https://api.adzuna.com/v1/api/jobs/in/search/1", params=params)
        for j in data.get("results", []):
            jid = str(j.get("id"))
            if jid in jobs:
                continue
            location = (j.get("location") or {}).get("display_name", "")
            predicted = str(j.get("salary_is_predicted")) == "1"
            salary = format_salary(j.get("salary_min"), j.get("salary_max"), "INR", "year")
            jobs[jid] = make_job(
                "adzuna", jid,
                title=html_to_text(j.get("title")),
                company=(j.get("company") or {}).get("display_name"),
                location=location,
                remote=looks_remote(j.get("title"), location, j.get("description")),
                url=j.get("redirect_url", ""),
                posted_at=iso_date(j.get("created")),
                salary=f"{salary} (estimated)" if salary and predicted else salary,
                employment_type=(j.get("contract_time") or "").replace("_", " ").title(),
                tags=[(j.get("category") or {}).get("label")],
                description=html_to_text(j.get("description")),
            )
    return list(jobs.values())[: ctx.per_source_limit]


# --------------------------------------------------------------------------- #
# JSearch (RapidAPI) — aggregates Google Jobs / LinkedIn / Naukri / Indeed.
# The old /search endpoint was retired (404); /search-v2 nests results in
# data.jobs and returns 10 per call. `country` defaults to "us" if omitted.
# --------------------------------------------------------------------------- #
JSEARCH_URL = "https://jsearch.p.rapidapi.com/search-v2"


def _record_rapidapi_quota(ctx: SearchContext, response: httpx.Response) -> None:
    if ctx.quota is None:
        return
    remaining = response.headers.get("x-ratelimit-requests-remaining")
    reset = response.headers.get("x-ratelimit-requests-reset")  # seconds until the plan period resets
    if remaining is not None and remaining.lstrip("-").isdigit():
        ctx.quota.update("jsearch", remaining=max(0, int(remaining)),
                         reset_in_seconds=float(reset) if reset and reset.isdigit() else None)
    else:
        ctx.quota.update("jsearch", used=1)


async def fetch_jsearch(ctx: SearchContext, client: httpx.AsyncClient) -> list[JobListing]:
    if not ctx.keys.rapidapi_key:
        raise SourceSkipped("Set RAPIDAPI_KEY to enable")
    headers = {"X-RapidAPI-Key": ctx.keys.rapidapi_key, "X-RapidAPI-Host": "jsearch.p.rapidapi.com"}
    jobs: dict[str, JobListing] = {}
    for query, remote in _guarded_plan(ctx, "jsearch"):
        params = {"country": "in", "date_posted": "month"}
        if remote:
            params |= {"query": f"{query} in India", "work_from_home": "true"}
        else:
            params["query"] = f"{query} in {ctx.city}, India"
        response = await client.get(JSEARCH_URL, params=params, headers=headers)
        _record_rapidapi_quota(ctx, response)
        response.raise_for_status()
        data = response.json()
        payload = data.get("data") or []
        results = payload.get("jobs", []) if isinstance(payload, dict) else payload  # v2 vs v1 shape
        for j in results:
            jid = j.get("job_id")
            if not jid or jid in jobs:
                continue
            location = j.get("job_location") or ", ".join(
                p for p in (j.get("job_city"), j.get("job_state"), j.get("job_country")) if p)
            is_remote = bool(j.get("job_is_remote"))
            publisher = j.get("job_publisher") or ""
            jobs[jid] = make_job(
                "jsearch", jid,
                title=j.get("job_title"),
                company=j.get("employer_name"),
                location=f"Remote ({location})" if is_remote and location else location,
                remote=is_remote,
                url=j.get("job_apply_link") or j.get("job_google_link") or "",
                posted_at=iso_date(j.get("job_posted_at_datetime_utc")),
                salary=j.get("job_salary_string") or format_salary(
                    j.get("job_min_salary"), j.get("job_max_salary"),
                    j.get("job_salary_currency"), (j.get("job_salary_period") or "").lower()),
                employment_type=", ".join(
                    t.replace("FULLTIME", "Full Time").replace("PARTTIME", "Part Time").title()
                    for t in j.get("job_employment_types") or []
                ) or (j.get("job_employment_type") or "").replace("–", "-"),
                tags=[f"via {publisher}" if publisher else ""],
                description=html_to_text(j.get("job_description")),
            )
    return list(jobs.values())[: ctx.per_source_limit]


# --------------------------------------------------------------------------- #
# SerpAPI — Google Jobs
# --------------------------------------------------------------------------- #
_GOOGLE_LOCATIONS = {"chennai": "Chennai, Tamil Nadu, India"}


async def _sync_serpapi_quota(ctx: SearchContext, client: httpx.AsyncClient) -> None:
    """Refresh remaining searches from SerpAPI's account endpoint (free, not metered)."""
    if ctx.quota is None:
        return
    try:
        account = await get_json(client, "https://serpapi.com/account.json",
                                 params={"api_key": ctx.keys.serpapi_key})
    except Exception:  # best effort; fall back to our own count
        return
    left = account.get("total_searches_left", account.get("plan_searches_left"))
    if isinstance(left, int):
        ctx.quota.update("serpapi", remaining=left, reset_at=_next_month_start())


def _next_month_start() -> str:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    first = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    nxt = first.replace(year=now.year + 1, month=1) if now.month == 12 else first.replace(month=now.month + 1)
    return nxt.isoformat(timespec="seconds")


async def fetch_serpapi(ctx: SearchContext, client: httpx.AsyncClient) -> list[JobListing]:
    if not ctx.keys.serpapi_key:
        raise SourceSkipped("Set SERPAPI_KEY to enable")
    await _sync_serpapi_quota(ctx, client)
    jobs: dict[str, JobListing] = {}
    for query, remote in _guarded_plan(ctx, "serpapi"):
        params = {
            "engine": "google_jobs",
            "q": f"{query} remote" if remote else query,
            "location": "India" if remote else _GOOGLE_LOCATIONS.get(ctx.city.lower(), f"{ctx.city}, India"),
            "gl": "in",
            "hl": "en",
            "api_key": ctx.keys.serpapi_key,
        }
        data = await get_json(client, "https://serpapi.com/search.json", params=params)
        if ctx.quota is not None:
            ctx.quota.update("serpapi", used=1)
        if error := data.get("error"):
            if "hasn't returned any results" in error:
                continue
            raise RuntimeError(f"SerpAPI: {error}")
        for j in data.get("jobs_results", []):
            jid = j.get("job_id")
            if not jid or jid in jobs:
                continue
            ext = j.get("detected_extensions") or {}
            location = j.get("location") or ""
            is_remote = bool(ext.get("work_from_home")) or looks_remote(location)
            apply = j.get("apply_options") or []
            jobs[jid] = make_job(
                "serpapi", jid[:120],
                title=j.get("title"),
                company=j.get("company_name"),
                location=f"Remote ({location})" if is_remote and location and "remote" not in location.lower() else location,
                remote=is_remote,
                url=(apply[0].get("link") if apply else "") or j.get("source_link") or j.get("share_link", ""),
                posted_at=iso_date(ext.get("posted_at")),
                salary=ext.get("salary", ""),
                employment_type=(ext.get("schedule_type") or "").replace("–", "-"),
                tags=[j.get("via", "")],
                description=html_to_text(j.get("description")),
            )
    return list(jobs.values())[: ctx.per_source_limit]
