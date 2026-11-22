"""Company career boards hosted on Greenhouse, Lever and Ashby (public, no key)."""
from __future__ import annotations

import asyncio

import httpx

from app.models import ATSBoardConfig, JobListing
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

BOARD_CACHE_TTL = 3600
MAX_CONCURRENT_BOARDS = 6


def _company(board: ATSBoardConfig) -> str:
    return board.label or board.slug.replace("-", " ").title()


async def _greenhouse(board: ATSBoardConfig, client: httpx.AsyncClient) -> list[JobListing]:
    data = await get_json(client, f"https://boards-api.greenhouse.io/v1/boards/{board.slug}/jobs",
                          params={"content": "true"}, cache_ttl=BOARD_CACHE_TTL)
    jobs = []
    for j in data.get("jobs", []):
        location = (j.get("location") or {}).get("name", "")
        jobs.append(make_job(
            "greenhouse", f"{board.slug}:{j['id']}",
            title=j.get("title"),
            company=j.get("company_name") or _company(board),
            location=location,
            remote=looks_remote(location),
            url=j.get("absolute_url", ""),
            posted_at=iso_date(j.get("first_published") or j.get("updated_at")),
            tags=[d.get("name") for d in j.get("departments") or []],
            description=html_to_text(j.get("content")),
        ))
    return jobs


async def _lever(board: ATSBoardConfig, client: httpx.AsyncClient) -> list[JobListing]:
    data = await get_json(client, f"https://api.lever.co/v0/postings/{board.slug}",
                          params={"mode": "json"}, cache_ttl=BOARD_CACHE_TTL)
    jobs = []
    for j in data if isinstance(data, list) else []:
        cats = j.get("categories") or {}
        locations = cats.get("allLocations") or [cats.get("location") or ""]
        location = " / ".join(loc for loc in locations if loc)
        salary = j.get("salaryRange") or {}
        jobs.append(make_job(
            "lever", f"{board.slug}:{j['id']}",
            title=j.get("text"),
            company=_company(board),
            location=location,
            remote=j.get("workplaceType") == "remote" or looks_remote(location),
            url=j.get("hostedUrl", ""),
            posted_at=iso_date(j.get("createdAt")),
            salary=format_salary(salary.get("min"), salary.get("max"), salary.get("currency"), salary.get("interval")),
            employment_type=cats.get("commitment") or "",
            tags=[cats.get("team"), cats.get("department"), j.get("workplaceType")],
            description=(j.get("descriptionPlain") or "") + "\n" + (j.get("additionalPlain") or ""),
        ))
    return jobs


async def _ashby(board: ATSBoardConfig, client: httpx.AsyncClient) -> list[JobListing]:
    data = await get_json(client, f"https://api.ashbyhq.com/posting-api/job-board/{board.slug}",
                          params={"includeCompensation": "true"}, cache_ttl=BOARD_CACHE_TTL)
    jobs = []
    for j in data.get("jobs", []):
        if j.get("isListed") is False:
            continue
        locations = [j.get("location") or ""] + [s.get("location", "") for s in j.get("secondaryLocations") or []]
        location = " / ".join(dict.fromkeys(loc for loc in locations if loc))
        workplace = (j.get("workplaceType") or "").lower()
        remote = workplace == "remote" if workplace else bool(j.get("isRemote"))
        comp = j.get("compensation") or {}
        jobs.append(make_job(
            "ashby", f"{board.slug}:{j['id']}",
            title=j.get("title"),
            company=_company(board),
            location=location,
            remote=remote or looks_remote(location),
            url=j.get("jobUrl", ""),
            posted_at=iso_date(j.get("publishedAt")),
            salary=comp.get("scrapeableCompensationSalarySummary") or "",
            employment_type=j.get("employmentType") or "",
            tags=[j.get("department"), j.get("team"), j.get("workplaceType")],
            description=j.get("descriptionPlain") or html_to_text(j.get("descriptionHtml")),
        ))
    return jobs


_PROVIDERS = {"greenhouse": _greenhouse, "lever": _lever, "ashby": _ashby}


async def fetch_ats_boards(ctx: SearchContext, client: httpx.AsyncClient) -> tuple[list[JobListing], str]:
    boards = [b for b in ctx.ats_boards if b.enabled]
    if not boards:
        raise SourceSkipped("No ATS boards enabled")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BOARDS)

    async def one(board: ATSBoardConfig) -> list[JobListing]:
        async with semaphore:
            return await _PROVIDERS[board.provider](board, client)

    results = await asyncio.gather(*(one(b) for b in boards), return_exceptions=True)

    jobs: list[JobListing] = []
    failures: list[str] = []
    for board, result in zip(boards, results):
        if isinstance(result, BaseException):
            reason = (f"HTTP {result.response.status_code}" if isinstance(result, httpx.HTTPStatusError)
                      else type(result).__name__)
            failures.append(f"{board.label or board.slug} ({reason})")
            continue
        jobs.extend(j for j in result if ctx.matcher.matches(j.title))

    if len(failures) == len(boards):
        raise RuntimeError(f"All {len(boards)} boards failed: {', '.join(failures)}")
    message = f"{len(boards) - len(failures)}/{len(boards)} boards ok"
    if failures:
        message += f"; failed: {', '.join(failures)}"
    # Boards return every opening worldwide; keep local/India/remote roles first
    # so the per-source cap never drops them in favour of e.g. US-only roles.
    city = ctx.city.lower()

    def priority(job: JobListing) -> int:
        loc = job.location.lower()
        if city and city in loc:
            return 0
        if "india" in loc:
            return 1
        return 2 if job.remote else 3

    jobs.sort(key=priority)
    return jobs[: ctx.per_source_limit], message
