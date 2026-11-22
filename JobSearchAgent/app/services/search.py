"""End-to-end search pipeline used by the API and the scheduler.

fetch (all sources, concurrent) -> refine (location filter + de-dup)
-> score (local + Gemini top-N) -> annotate (new / tracked, hide dismissed)
-> cap to max_results -> mark seen -> save to history
"""
from __future__ import annotations

import logging
from typing import Literal

from app.models import SearchRequest, SearchResponse
from app.services.filtering import refine_jobs
from app.services.scoring import score_jobs
from app.services.storage import Storage, get_storage
from app.sources import build_context, fetch_all
from app.sources.quota import QuotaTracker

log = logging.getLogger(__name__)


async def run_search(
    request: SearchRequest,
    storage: Storage | None = None,
    trigger: Literal["manual", "scheduled"] = "manual",
) -> SearchResponse:
    """Run a full search. Raises ValueError if there's nothing to search for."""
    storage = storage or get_storage()
    # Validates queries, resolves keys (request overrides .env), attaches the quota guard.
    ctx = build_context(request, quota=QuotaTracker(storage))

    raw_jobs, statuses = await fetch_all(request, ctx=ctx)
    refined = refine_jobs(raw_jobs, city=ctx.city, remote_only=request.remote_only)
    scored = await score_jobs(refined.jobs, request.profile, ctx.keys.gemini_api_key, ctx.keys.gemini_model)
    annotated = storage.annotate(scored.jobs)

    shown = annotated.jobs[: request.max_results]
    storage.mark_seen(shown)

    excluded = dict(refined.excluded)
    if annotated.hidden_count:
        excluded["dismissed"] = annotated.hidden_count

    response = SearchResponse(
        total_fetched=len(raw_jobs),
        total_returned=len(shown),
        scoring_engine=scored.engine,
        sources=statuses,
        excluded=excluded,
        duplicates_merged=refined.duplicates_merged,
        jobs=shown,
        new_count=sum(j.is_new for j in shown),
    )
    summary = storage.save_search(response, location=request.location, queries=ctx.queries, trigger=trigger)
    log.info("Search %s: %d fetched -> %d shown (%d new), engine=%s",
             summary.id, len(raw_jobs), len(shown), response.new_count, scored.engine)
    return response.model_copy(update={"search_id": summary.id})
