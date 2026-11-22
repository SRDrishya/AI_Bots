"""Free, no-key job sources. Mostly remote roles."""
from __future__ import annotations

import httpx

from app.models import JobListing
from app.sources.base import (
    SearchContext,
    format_salary,
    get_json,
    html_to_text,
    iso_date,
    looks_remote,
    make_job,
)

HOUR = 3600


# --------------------------------------------------------------------------- #
# Remotive — remote only. Their API notice asks for infrequent polling, so we
# pull the full list once (cached 6h) and filter titles locally; their own
# `search` param is too loose to be useful.
# --------------------------------------------------------------------------- #
async def fetch_remotive(ctx: SearchContext, client: httpx.AsyncClient) -> list[JobListing]:
    data = await get_json(client, "https://remotive.com/api/remote-jobs", cache_ttl=6 * HOUR)
    jobs: list[JobListing] = []
    for j in data.get("jobs", []):
        if not ctx.matcher.matches(j.get("title", "")):
            continue
        region = (j.get("candidate_required_location") or "Worldwide").strip()
        jobs.append(make_job(
            "remotive", j["id"],
            title=j.get("title"),
            company=j.get("company_name"),
            location=f"Remote ({region})",
            remote=True,
            url=j.get("url", ""),
            posted_at=iso_date(j.get("publication_date")),
            salary=j.get("salary") or "",
            employment_type=(j.get("job_type") or "").replace("_", " ").title(),
            tags=[j.get("category"), *(j.get("tags") or [])],
            description=html_to_text(j.get("description")),
        ))
    return jobs[: ctx.per_source_limit]


# --------------------------------------------------------------------------- #
# Arbeitnow — Europe-centric board; we keep only remote roles.
# --------------------------------------------------------------------------- #
async def fetch_arbeitnow(ctx: SearchContext, client: httpx.AsyncClient) -> list[JobListing]:
    jobs: list[JobListing] = []
    for page in (1, 2, 3):
        data = await get_json(client, "https://www.arbeitnow.com/api/job-board-api",
                              params={"page": page}, cache_ttl=HOUR)
        for j in data.get("data", []):
            remote = bool(j.get("remote")) or looks_remote(j.get("location"))
            if not remote or not ctx.matcher.matches(j.get("title", "")):
                continue
            location = j.get("location") or ""
            jobs.append(make_job(
                "arbeitnow", j.get("slug") or j.get("url"),
                title=j.get("title"),
                company=j.get("company_name"),
                location=f"Remote ({location})" if location and "remote" not in location.lower() else location or "Remote",
                remote=True,
                url=j.get("url", ""),
                posted_at=iso_date(j.get("created_at")),
                employment_type=", ".join(j.get("job_types") or []),
                tags=j.get("tags") or [],
                description=html_to_text(j.get("description")),
            ))
        if not (data.get("links") or {}).get("next"):
            break
    return jobs[: ctx.per_source_limit]


# --------------------------------------------------------------------------- #
# Himalayas — remote jobs; `country=India` returns roles open to India-based
# candidates (including worldwide ones).
# --------------------------------------------------------------------------- #
def _himalayas_location(restrictions: list[str]) -> str:
    if not restrictions:
        return "Remote (Worldwide)"
    if "India" in restrictions and len(restrictions) > 3:
        return f"Remote (India + {len(restrictions) - 1} more countries)"
    if len(restrictions) > 3:
        return f"Remote ({', '.join(restrictions[:3])} + {len(restrictions) - 3} more)"
    return f"Remote ({', '.join(restrictions)})"


async def fetch_himalayas(ctx: SearchContext, client: httpx.AsyncClient) -> list[JobListing]:
    jobs: dict[str, JobListing] = {}
    for query in ctx.queries[:3]:
        data = await get_json(client, "https://himalayas.app/jobs/api/search",
                              params={"q": query, "country": "India"}, cache_ttl=HOUR)
        for j in data.get("jobs", []):
            guid = j.get("guid") or j.get("applicationLink")
            if not guid or guid in jobs:
                continue
            jobs[guid] = make_job(
                "himalayas", guid.rstrip("/").rsplit("/", 1)[-1],
                title=j.get("title"),
                company=j.get("companyName"),
                location=_himalayas_location(j.get("locationRestrictions") or []),
                remote=True,
                url=j.get("applicationLink") or guid,
                posted_at=iso_date(j.get("pubDate")),
                salary=format_salary(j.get("minSalary"), j.get("maxSalary"), j.get("currency"), j.get("salaryPeriod")),
                employment_type=j.get("employmentType") or "",
                tags=[*(j.get("seniority") or []), *(c.replace("-", " ") for c in j.get("categories") or [])],
                description=html_to_text(j.get("description")) or j.get("excerpt", ""),
            )
    return list(jobs.values())[: ctx.per_source_limit]


# --------------------------------------------------------------------------- #
# Jobicy — remote only; no India geo filter, so jobGeo is kept in `location`
# for the location filter to judge (Anywhere / APAC are useful from Chennai).
# Their `tag` search is loose, so titles are filtered locally as well.
# --------------------------------------------------------------------------- #
async def fetch_jobicy(ctx: SearchContext, client: httpx.AsyncClient) -> list[JobListing]:
    jobs: dict[int, JobListing] = {}
    for query in ctx.queries[:2]:
        data = await get_json(client, "https://jobicy.com/api/v2/remote-jobs",
                              params={"count": 50, "tag": query}, cache_ttl=HOUR)
        for j in data.get("jobs", []):
            title = html_to_text(j.get("jobTitle"))
            if j["id"] in jobs or not ctx.matcher.matches(title):  # their tag search is loose
                continue
            jobs[j["id"]] = make_job(
                "jobicy", j["id"],
                title=title,
                company=j.get("companyName"),
                location=f"Remote ({j.get('jobGeo') or 'Anywhere'})",
                remote=True,
                url=j.get("url", ""),
                posted_at=iso_date(j.get("pubDate")),
                salary=format_salary(j.get("salaryMin") or j.get("annualSalaryMin"),
                                     j.get("salaryMax") or j.get("annualSalaryMax"),
                                     j.get("salaryCurrency"), j.get("salaryPeriod")),
                employment_type=", ".join(j.get("jobType") or []),
                tags=[j.get("jobLevel"), *(j.get("jobIndustry") or [])],
                description=html_to_text(j.get("jobDescription")) or j.get("jobExcerpt", ""),
            )
    return list(jobs.values())[: ctx.per_source_limit]
