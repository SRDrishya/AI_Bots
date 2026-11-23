"""FastAPI app: JSON API under /api plus the single-page UI at /."""
from __future__ import annotations

import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import BASE_DIR, MAX_RESUME_BYTES, get_env_settings
from app.models import (
    ApplicationKit,
    ApplicationKitRequest,
    CombinedSearch,
    EmailAccountsResponse,
    EmailJobsResponse,
    CompanyInsightsRequest,
    CompanyInsightsResponse,
    CompanyLink,
    ResumeDocument,
    ResumeFileInfo,
    ResumeProfile,
    ResumeUploadResult,
    SearchOptions,
    SearchRequest,
    SearchResponse,
    SearchSummary,
    StoredSearch,
    TailoredResume,
    TailorRequest,
    TrackedJob,
    TrackingStatus,
    TrackJobRequest,
    TrackUpdateRequest,
)
from app.services import export
from app.services.coaching import KitCache, generate_kit
from app.services.company import InsightCache, generate_insight, quick_links
from app.services.email_jobs import EmailIntegrationError, EmailJobService
from app.services.ai import NO_KEY_HINT, ai_available, describe as describe_ai
from app.services.gemini import GeminiError
from app.services.resume import ResumeParseError, analyze_text, extract_text
from app.services.scheduler import ScheduleStatus, SearchScheduler
from app.services.search import run_search
from app.services.storage import Storage, get_storage, job_key
from app.services.tailor import TailoredStore, prepare_tailored, refresh, render as render_tailored
from app.sources import SOURCES
from app.sources.base import resolve_keys
from app.sources.quota import QuotaTracker

log = logging.getLogger(__name__)
STATIC_DIR = BASE_DIR / "static"
KEYED_SOURCES = {"adzuna": ("adzuna_app_id", "adzuna_app_key"), "jsearch": ("rapidapi_key",),
                 "serpapi": ("serpapi_key",)}


class SourceInfo(BaseModel):
    name: str
    needs_key: bool
    configured: bool


class AppStatus(BaseModel):
    has_profile: bool
    gemini_configured: bool
    gemini_model: str
    ai_providers: str = ""  # e.g. 'Claude (claude-opus-5-5), fallback Gemini (gemini-3.8-flash)'
    sources: list[SourceInfo]
    quota: dict
    schedule: ScheduleStatus
    tracker: dict[str, int]


class ScheduleUpdate(BaseModel):
    enabled: bool | None = None
    run_time: str | None = None
    location: str | None = None
    remote_only: bool | None = None
    max_results: int | None = None


def create_app(storage: Storage | None = None, start_scheduler: bool = True) -> FastAPI:
    store = storage or get_storage()
    settings = get_env_settings()
    email_jobs = EmailJobService(
        store,
        google_client_id=settings.google_oauth_client_id,
        google_client_secret=settings.google_oauth_client_secret,
        microsoft_client_id=settings.microsoft_oauth_client_id,
        microsoft_client_secret=settings.microsoft_oauth_client_secret,
        redirect_base=settings.email_oauth_redirect_base,
    )
    scheduler = SearchScheduler(store, email_scan=email_jobs.scan)
    background: set[asyncio.Task] = set()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if start_scheduler:
            scheduler.start()
        yield
        await scheduler.stop()
        for task in background:
            task.cancel()

    app = FastAPI(title="Job Search Agent", version="0.1.0", lifespan=lifespan)
    app.state.storage = store
    app.state.scheduler = scheduler
    app.state.email_jobs = email_jobs

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # ------------------------------------------------------------------ status
    @app.get("/api/status", response_model=AppStatus)
    def status() -> AppStatus:
        keys = resolve_keys(None)
        sources = [
            SourceInfo(name=name, needs_key=name in KEYED_SOURCES,
                       configured=all(getattr(keys, f) for f in KEYED_SOURCES.get(name, ())))
            for name in SOURCES
        ]
        return AppStatus(
            has_profile=store.load_profile() is not None,
            gemini_configured=ai_available(keys.gemini_api_key),
            gemini_model=keys.gemini_model,
            ai_providers=describe_ai(keys.gemini_api_key, keys.gemini_model),
            sources=sources,
            quota=QuotaTracker(store).snapshot(),
            schedule=scheduler.status(),
            tracker=store.tracker_stats(),
        )

    # ------------------------------------------------------------------ email job opportunities
    @app.get("/api/emails/accounts", response_model=EmailAccountsResponse)
    def email_accounts() -> EmailAccountsResponse:
        try:
            return email_jobs.accounts()
        except EmailIntegrationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/emails/connect/{provider}")
    def connect_email(provider: str) -> RedirectResponse:
        try:
            return RedirectResponse(email_jobs.begin_auth(provider), status_code=302)
        except EmailIntegrationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/emails/oauth/{provider}/callback")
    async def email_oauth_callback(
        provider: str,
        code: str | None = None,
        state: str | None = None,
        error: str | None = None,
    ) -> RedirectResponse:
        if error:
            message = "Email sign-in was cancelled or denied"
        elif not code or not state:
            message = "Email sign-in did not return the required authorization details"
        else:
            try:
                await email_jobs.finish_auth(provider, code, state)
                return RedirectResponse("/?email_connected=1#email", status_code=303)
            except EmailIntegrationError as exc:
                message = str(exc)
        return RedirectResponse(f"/?{urlencode({'email_error': message[:240]})}#email", status_code=303)

    @app.delete("/api/emails/accounts/{provider}", status_code=204)
    async def disconnect_email(provider: str) -> Response:
        try:
            await email_jobs.disconnect(provider)
        except EmailIntegrationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(status_code=204)

    @app.get("/api/emails/jobs", response_model=EmailJobsResponse)
    def email_job_opportunities() -> EmailJobsResponse:
        """Job emails saved by earlier scans (no mailbox access)."""
        return email_jobs.saved_opportunities()

    @app.post("/api/emails/scan", response_model=EmailJobsResponse)
    async def scan_email_jobs(
        full: bool = Query(default=False, description="Ignore the last scan time and look back `days`"),
        days: int = Query(default=7, ge=1, le=365, description="Window for first/full scans"),
        limit: int = Query(default=100, ge=1, le=200),
    ) -> EmailJobsResponse:
        # Only mail since each account's previous scan is fetched. Per-account failures come back
        # in `errors` so one broken inbox doesn't hide the other.
        return await email_jobs.scan(full=full, days=days, limit=limit)

    # ------------------------------------------------------------------ resume / profile
    @app.post("/api/resume", response_model=ResumeUploadResult)
    async def upload_resume(file: UploadFile = File(...)) -> ResumeUploadResult:
        """Analyse a resume once. Re-uploading the same file keeps the saved profile
        (and any edits) without calling the AI again."""
        data = await file.read(MAX_RESUME_BYTES + 1)
        try:
            text = extract_text(file.filename or "", data)
        except ResumeParseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        saved_info, saved_profile = store.resume_file_info(), store.load_profile()
        if saved_info and saved_profile and saved_info.sha256 == hashlib.sha256(data).hexdigest():
            return ResumeUploadResult(profile=saved_profile, file=saved_info, reused=True)

        keys = resolve_keys(None)
        profile = await analyze_text(text, keys.gemini_api_key, keys.gemini_model)
        store.save_profile(profile)
        store.save_resume_text(text)
        info = store.save_resume_file(file.filename or "resume", data)
        return ResumeUploadResult(profile=profile, file=info)

    @app.post("/api/resume/reanalyze", response_model=ResumeUploadResult)
    async def reanalyze_resume() -> ResumeUploadResult:
        """Explicitly re-run the analysis on the saved resume (overwrites profile edits)."""
        text = store.load_resume_text()
        if not text:
            raise HTTPException(status_code=404, detail="No saved resume — upload one first")
        keys = resolve_keys(None)
        profile = await analyze_text(text, keys.gemini_api_key, keys.gemini_model)
        store.save_profile(profile)
        return ResumeUploadResult(profile=profile, file=store.resume_file_info())

    @app.get("/api/resume", response_model=ResumeFileInfo)
    def resume_info() -> ResumeFileInfo:
        info = store.resume_file_info()
        if info is None:
            raise HTTPException(status_code=404, detail="No resume file saved")
        return info

    @app.get("/api/resume/file")
    def resume_file() -> FileResponse:
        path = store.resume_file_path()
        if path is None:
            raise HTTPException(status_code=404, detail="No resume file saved")
        return FileResponse(path, filename=path.name)

    @app.get("/api/profile", response_model=ResumeProfile)
    def get_profile() -> ResumeProfile:
        profile = store.load_profile()
        if profile is None:
            raise HTTPException(status_code=404, detail="No resume uploaded yet")
        return profile

    @app.put("/api/profile", response_model=ResumeProfile)
    def put_profile(profile: ResumeProfile) -> ResumeProfile:
        store.save_profile(profile)
        return profile

    # ------------------------------------------------------------------ search
    @app.post("/api/search", response_model=SearchResponse)
    async def search(options: SearchOptions) -> SearchResponse:
        profile = store.load_profile()
        if profile is None:
            raise HTTPException(status_code=400, detail="Upload your resume first")
        request = SearchRequest(profile=profile, **options.model_dump(exclude_none=True))
        return await run_search(request, storage=store, trigger="manual")

    @app.get("/api/searches", response_model=list[SearchSummary])
    def searches() -> list[SearchSummary]:
        return store.list_searches()

    # Declared before /api/searches/{search_id} so "combined" isn't taken for a search id.
    @app.get("/api/searches/combined", response_model=CombinedSearch)
    def combined_searches(days: int = Query(3, ge=1, le=90)) -> CombinedSearch:
        """All saved searches from the last `days` calendar days (IST), duplicates removed."""
        return store.combine_searches(days)

    @app.get("/api/searches/combined/export")
    def export_combined(days: int = Query(3, ge=1, le=90), format: str = "xlsx") -> Response:
        fmt = _check_format(format)
        combined = store.combine_searches(days)
        jobs = combined.response.jobs
        stem = f"job-matches-last-{days}-days-{date.today():%Y-%m-%d}"
        if fmt == "csv":
            return _download(export.jobs_csv(jobs), fmt, stem)
        meta = {"View": f"Last {days} day(s), combined", "From": combined.start[:10],
                "Searches included": str(len(combined.searches)),
                "Listings before de-dup": str(combined.total_listings),
                "Duplicates removed": str(combined.duplicates_removed)}
        return _download(export.jobs_xlsx(jobs, store.load_profile(), meta), fmt, stem)

    @app.get("/api/searches/latest", response_model=StoredSearch)
    def latest_search() -> StoredSearch:
        stored = store.latest_search()
        if stored is None:
            raise HTTPException(status_code=404, detail="No searches yet")
        return stored

    @app.get("/api/searches/{search_id}", response_model=StoredSearch)
    def get_search(search_id: str) -> StoredSearch:
        stored = store.load_search(search_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="Search not found")
        return stored

    # ------------------------------------------------------------------ tracker
    @app.get("/api/tracker", response_model=list[TrackedJob])
    def tracker(status: TrackingStatus | None = None) -> list[TrackedJob]:
        return store.list_tracked(status)

    @app.post("/api/tracker", response_model=TrackedJob)
    def track(body: TrackJobRequest) -> TrackedJob:
        return store.track(body.job, body.status, body.notes)

    @app.patch("/api/tracker/{key}", response_model=TrackedJob)
    def update_tracked(key: str, body: TrackUpdateRequest) -> TrackedJob:
        try:
            return store.update_tracked(key, status=body.status, notes=body.notes)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job is not tracked") from exc

    @app.delete("/api/tracker/{key}", status_code=204)
    def untrack(key: str) -> None:
        if not store.untrack(key):
            raise HTTPException(status_code=404, detail="Job is not tracked")

    # ------------------------------------------------------------------ export
    def _download(content: bytes, fmt: str, stem: str) -> Response:
        media = ("text/csv; charset=utf-8" if fmt == "csv"
                 else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        return Response(content, media_type=media,
                        headers={"Content-Disposition": f'attachment; filename="{stem}.{fmt}"'})

    def _check_format(fmt: str) -> str:
        if fmt not in ("csv", "xlsx"):
            raise HTTPException(status_code=400, detail="format must be csv or xlsx")
        return fmt

    @app.get("/api/searches/{search_id}/export")
    def export_search(search_id: str, format: str = "xlsx") -> Response:
        fmt = _check_format(format)
        stored = store.latest_search() if search_id == "latest" else store.load_search(search_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="Search not found")
        jobs, summary = stored.response.jobs, stored.summary
        stem = f"job-matches-{summary.created_at[:10]}"
        if fmt == "csv":
            return _download(export.jobs_csv(jobs), fmt, stem)
        meta = {"Search run": summary.created_at, "Type": summary.trigger, "Location": summary.location,
                "Queries": ", ".join(summary.queries), "Scoring": summary.scoring_engine}
        return _download(export.jobs_xlsx(jobs, store.load_profile(), meta), fmt, stem)

    @app.get("/api/tracker/export")
    def export_tracker(format: str = "xlsx") -> Response:
        fmt = _check_format(format)
        items = store.list_tracked()
        content = export.tracker_csv(items) if fmt == "csv" else export.tracker_xlsx(items)
        return _download(content, fmt, f"applications-{date.today():%Y-%m-%d}")

    # ------------------------------------------------------------------ application kit
    @app.post("/api/application-kit", response_model=ApplicationKit)
    async def application_kit(body: ApplicationKitRequest) -> ApplicationKit:
        keys = resolve_keys(None)
        if not ai_available(keys.gemini_api_key):
            raise HTTPException(status_code=400, detail=f"Cover letters need {NO_KEY_HINT}")
        profile = store.load_profile()
        if profile is None:
            raise HTTPException(status_code=400, detail="Upload your resume first")
        job = body.job if body.job.key else body.job.model_copy(update={"key": job_key(body.job)})
        cache = KitCache(store)
        if not body.regenerate and (cached := cache.get(job.key, body.tone)):
            return cached
        try:
            kit = await generate_kit(job, profile, store.load_resume_text(), api_key=keys.gemini_api_key,
                                     model=keys.gemini_model, tone=body.tone)
        except GeminiError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        cache.put(kit)
        return kit

    # ------------------------------------------------------------------ company insights
    # ------------------------------------------------------------------ tailored resume
    tailored = TailoredStore(store)

    def _tailored_or_404(key: str) -> TailoredResume:
        draft = tailored.get(key) if TailoredStore.valid_key(key) else None
        if draft is None:
            raise HTTPException(status_code=404, detail="No tailored resume for this job yet")
        return draft

    @app.get("/api/tailored-resumes", response_model=list[TailoredResume])
    def tailored_list() -> list[TailoredResume]:
        return tailored.list()

    @app.post("/api/tailored-resume", response_model=TailoredResume)
    async def tailored_prepare(body: TailorRequest) -> TailoredResume:
        """AI draft of a resume tailored to one job (cached per job unless regenerate=true)."""
        keys = resolve_keys(None)
        if not ai_available(keys.gemini_api_key):
            raise HTTPException(status_code=400, detail=f"Tailored resumes need {NO_KEY_HINT}")
        profile, resume_text = store.load_profile(), store.load_resume_text()
        if profile is None or not resume_text.strip():
            raise HTTPException(status_code=400, detail="Upload your original resume first")
        job = body.job if body.job.key else body.job.model_copy(update={"key": job_key(body.job)})
        if not body.regenerate and (cached := tailored.get(job.key)):
            return cached
        description = body.job_description.strip() or job.description
        try:
            draft = await prepare_tailored(job, description, profile, resume_text,
                                           api_key=keys.gemini_api_key, model=keys.gemini_model)
        except GeminiError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if previous := tailored.get(job.key):
            draft.files = previous.files  # files already generated stay downloadable
        tailored.put(draft)
        return draft

    @app.get("/api/tailored-resume/{key}", response_model=TailoredResume)
    def tailored_get(key: str) -> TailoredResume:
        return _tailored_or_404(key)

    @app.put("/api/tailored-resume/{key}", response_model=TailoredResume)
    def tailored_save(key: str, resume: ResumeDocument) -> TailoredResume:
        """Save edits from the preview; returns updated keyword coverage, warnings and page fit."""
        draft = refresh(_tailored_or_404(key).model_copy(update={"resume": resume}), store.load_resume_text())
        tailored.put(draft)
        return draft

    @app.post("/api/tailored-resume/{key}/generate")
    def tailored_generate(key: str, resume: ResumeDocument, format: str = "pdf") -> Response:
        """Save the previewed content, render it as PDF or DOCX, keep a copy and download it."""
        if format not in ("pdf", "docx"):
            raise HTTPException(status_code=400, detail="format must be pdf or docx")
        draft = refresh(_tailored_or_404(key).model_copy(update={"resume": resume}), store.load_resume_text())
        content, media = render_tailored(draft, format)
        draft = tailored.save_file(draft, format, content)
        filename = draft.files[-1].filename
        headers = {"Content-Disposition": f'attachment; filename="{filename}"',
                   "X-Page-Count": str(draft.fit.pages), "X-Font-Size": f"{draft.fit.font_size:g}"}
        return Response(content, media_type=media, headers=headers)

    @app.get("/api/tailored-resume/{key}/files/{filename}")
    def tailored_file(key: str, filename: str) -> FileResponse:
        path = tailored.file_path(key, filename) if TailoredStore.valid_key(key) else None
        if path is None:
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path, filename=filename)

    @app.get("/api/company/links", response_model=list[CompanyLink])
    def company_links(company: str, role: str = "", location: str = "Chennai") -> list[CompanyLink]:
        if not company.strip():
            raise HTTPException(status_code=400, detail="company is required")
        return quick_links(company, role, location)

    @app.post("/api/company/insights", response_model=CompanyInsightsResponse)
    async def company_insights(body: CompanyInsightsRequest) -> CompanyInsightsResponse:
        """Never fails because of the AI: links are always returned, `ai_error` explains a missing summary."""
        response = CompanyInsightsResponse(company=body.company,
                                           links=quick_links(body.company, body.role, body.location))
        keys = resolve_keys(None)
        if not ai_available(keys.gemini_api_key):
            response.ai_error = f"Add {NO_KEY_HINT} for an AI summary — the links above still work."
            return response
        cache = InsightCache(store)
        if not body.regenerate and (cached := cache.get(body.company, body.role)):
            response.insight = cached
            return response
        try:
            insight = await generate_insight(body.company, body.role, body.location,
                                             api_key=keys.gemini_api_key, model=keys.gemini_model)
        except GeminiError as exc:
            response.ai_error = str(exc)
            return response
        cache.put(insight)
        response.insight = insight
        return response

    # ------------------------------------------------------------------ schedule
    @app.get("/api/schedule", response_model=ScheduleStatus)
    def schedule() -> ScheduleStatus:
        return scheduler.status()

    @app.put("/api/schedule", response_model=ScheduleStatus)
    def update_schedule(body: ScheduleUpdate) -> ScheduleStatus:
        scheduler.update_settings(**body.model_dump(exclude_none=True))
        return scheduler.status()

    @app.post("/api/schedule/run", response_model=ScheduleStatus, status_code=202)
    async def run_schedule_now() -> ScheduleStatus:
        if not scheduler.status().running:
            task = asyncio.create_task(scheduler.run_now("run now"))
            background.add(task)
            task.add_done_callback(background.discard)
            await asyncio.sleep(0)  # let it take the lock so the response shows running=true
        return scheduler.status()

    # ------------------------------------------------------------------ UI
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(Path(STATIC_DIR) / "index.html")

    return app


app = create_app()
