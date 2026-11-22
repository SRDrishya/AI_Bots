"""Job source adapters and the aggregator that runs them concurrently."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import httpx

from app.config import STARTER_ATS_BOARDS
from app.models import ATSBoardConfig, JobListing, SearchRequest, SourceStatus
from app.sources.ats import fetch_ats_boards
from app.sources.base import SearchContext, SourceSkipped, TitleMatcher, build_queries, new_client, resolve_keys
from app.sources.free import fetch_arbeitnow, fetch_himalayas, fetch_jobicy, fetch_remotive
from app.sources.paid import fetch_adzuna, fetch_jsearch, fetch_serpapi
from app.sources.quota import QuotaTracker

log = logging.getLogger(__name__)

SourceFn = Callable[[SearchContext, httpx.AsyncClient], Awaitable[list[JobListing] | tuple[list[JobListing], str]]]

SOURCES: dict[str, SourceFn] = {
    "remotive": fetch_remotive,
    "arbeitnow": fetch_arbeitnow,
    "himalayas": fetch_himalayas,
    "jobicy": fetch_jobicy,
    "ats_boards": fetch_ats_boards,
    "adzuna": fetch_adzuna,
    "jsearch": fetch_jsearch,
    "serpapi": fetch_serpapi,
}
SOURCE_TIMEOUT = 45.0


def _friendly_error(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (401, 403):
            return f"API key rejected (HTTP {code})"
        if code == 429:
            return "Rate limit / quota reached (HTTP 429)"
        return f"HTTP {code} from {exc.request.url.host}"
    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException)):
        return "Timed out"
    if isinstance(exc, httpx.TransportError):
        return f"Network error: {type(exc).__name__}"
    return str(exc) or type(exc).__name__


def build_context(request: SearchRequest, quota: QuotaTracker | None = None) -> SearchContext:
    queries = build_queries(request.profile, request.custom_query)
    if not queries:
        raise ValueError("No search terms: upload a resume or enter a custom query.")
    boards = request.ats_boards or [ATSBoardConfig(**b) for b in STARTER_ATS_BOARDS]
    return SearchContext(
        queries=queries,
        location=request.location,
        remote_only=request.remote_only,
        keys=resolve_keys(request.api_keys),
        ats_boards=boards,
        # Fetch more than we return: later location filtering / dedup / scoring trims it.
        per_source_limit=min(max(request.max_results * 2, 50), 200),
        matcher=TitleMatcher(queries, request.profile.target_roles, request.profile.skills),
        quota=quota,
    )


async def _run(name: str, ctx: SearchContext, client: httpx.AsyncClient) -> tuple[list[JobListing], SourceStatus]:
    fn = SOURCES.get(name)
    if fn is None:
        return [], SourceStatus(source=name, status="error", message="Unknown source")
    try:
        result = await asyncio.wait_for(fn(ctx, client), timeout=SOURCE_TIMEOUT)
    except SourceSkipped as exc:
        return [], SourceStatus(source=name, status="skipped", message=str(exc))
    except Exception as exc:
        log.warning("Source %s failed: %r", name, exc)
        return [], SourceStatus(source=name, status="error", message=_friendly_error(exc))
    jobs, message = result if isinstance(result, tuple) else (result, "")
    return jobs, SourceStatus(source=name, status="ok", count=len(jobs), message=message)


async def fetch_all(
    request: SearchRequest,
    client: httpx.AsyncClient | None = None,
    ctx: SearchContext | None = None,
) -> tuple[list[JobListing], list[SourceStatus]]:
    """Run every requested source concurrently. Never raises for source failures."""
    ctx = ctx or build_context(request)
    names = list(dict.fromkeys(request.sources))
    owns_client = client is None
    client = client or new_client()
    try:
        results = await asyncio.gather(*(_run(n, ctx, client) for n in names))
    finally:
        if owns_client:
            await client.aclose()
    jobs = [job for source_jobs, _ in results for job in source_jobs]
    return jobs, [status for _, status in results]
