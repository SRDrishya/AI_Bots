"""JSON-file persistence: profile, search history, seen-jobs, application tracker.

Layout (under DATA_DIR, default ./data):
    profile.json            current ResumeProfile
    seen.json               {job_key: {"first_seen", "last_seen"}}  -> "new" badges
    tracker.json            {job_key: TrackedJob}
    searches/index.json     [SearchSummary, ...] newest first
    searches/<id>.json      StoredSearch (full results)

Writes are atomic (temp file + os.replace) and serialised with a lock.
A corrupted file is moved aside as <name>.corrupt-<timestamp> and treated as empty.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import IST, get_env_settings
from app.models import (
    CombinedSearch,
    JobLink,
    ResumeFileInfo,
    ResumeProfile,
    ScoredJob,
    SearchResponse,
    SearchSummary,
    StatusChange,
    StoredSearch,
    TrackedJob,
    TrackingStatus,
)
from app.services.filtering import normalize_company, normalize_title

log = logging.getLogger(__name__)

MAX_SEARCHES = 20
SEEN_RETENTION_DAYS = 90
HIDDEN_STATUSES = {"dismissed"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def job_key(job: ScoredJob) -> str:
    """Same opening from different sources -> same key (matches de-dup rules)."""
    company = normalize_company(job.company)
    title = normalize_title(job.title, company=company)
    return hashlib.sha1(f"{company}|{title}".encode()).hexdigest()[:16]


@dataclass
class AnnotateResult:
    jobs: list[ScoredJob]
    new_count: int
    hidden_count: int


class Storage:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ low level
    def _path(self, rel: str) -> Path:
        return self.root / rel

    def _read(self, rel: str, default: Any) -> Any:
        path = self._path(rel)
        with self._lock:
            if not path.exists():
                return default
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                backup = path.with_name(f"{path.name}.corrupt-{datetime.now():%Y%m%d%H%M%S}")
                path.replace(backup)
                log.error("Corrupted %s moved to %s (%s); starting fresh", path, backup.name, exc)
                return default

    def _write(self, rel: str, data: Any) -> None:
        path = self._path(rel)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=1)
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

    # ------------------------------------------------------------------ small documents
    def load_doc(self, name: str, default: Any) -> Any:
        """Generic small JSON document (quota.json, schedule.json, ...)."""
        return self._read(name, default)

    def save_doc(self, name: str, data: Any) -> None:
        self._write(name, data)

    # ------------------------------------------------------------------ profile
    def save_profile(self, profile: ResumeProfile) -> None:
        self._write("profile.json", {"saved_at": now_iso(), "profile": profile.model_dump()})

    # ------------------------------------------------------------------ original resume file
    RESUME_DIR = "resume"

    def save_resume_file(self, filename: str, data: bytes) -> ResumeFileInfo:
        """Keep the uploaded file in data/resume/ (replacing any previous one)."""
        name = Path(filename or "resume").name  # drop any directory parts
        stem, suffix = Path(name).stem, Path(name).suffix.lower()
        stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" ._") or "resume"
        safe_name = f"{stem[:80]}{suffix}"
        info = ResumeFileInfo(filename=safe_name, size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                              uploaded_at=now_iso())
        with self._lock:
            folder = self._path(self.RESUME_DIR)
            folder.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=folder, prefix=".upload.", suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                for old in folder.iterdir():  # only the current resume is kept
                    if old.is_file() and old.name != Path(tmp).name:
                        old.unlink()
                os.replace(tmp, folder / safe_name)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
            self._write("resume_file.json", info.model_dump())
        return info

    def resume_file_info(self) -> ResumeFileInfo | None:
        raw = self._read("resume_file.json", None)
        if not raw:
            return None
        try:
            info = ResumeFileInfo.model_validate(raw)
        except ValidationError:
            return None
        return info if self.resume_file_path(info) else None

    def resume_file_path(self, info: ResumeFileInfo | None = None) -> Path | None:
        info = info or self.resume_file_info()
        if info is None:
            return None
        path = self._path(self.RESUME_DIR) / info.filename
        return path if path.is_file() else None

    def save_resume_text(self, text: str) -> None:
        """Full resume text (local only) — used to write tailored cover letters."""
        self._write("resume_text.json", {"saved_at": now_iso(), "text": text})

    def load_resume_text(self) -> str:
        return (self._read("resume_text.json", {}) or {}).get("text", "")

    def load_profile(self) -> ResumeProfile | None:
        data = self._read("profile.json", None)
        if not data:
            return None
        try:
            return ResumeProfile.model_validate(data.get("profile", {}))
        except ValidationError:
            log.warning("profile.json didn't match the current schema; ignoring it")
            return None

    # ------------------------------------------------------------------ seen / annotate
    def annotate(self, jobs: list[ScoredJob]) -> AnnotateResult:
        """Attach key / is_new / first_seen / tracking_status and drop dismissed jobs.
        Read-only: call mark_seen() for the jobs actually shown to the user."""
        seen: dict[str, dict] = self._read("seen.json", {})
        tracker: dict[str, dict] = self._read("tracker.json", {})
        now = now_iso()
        out: list[ScoredJob] = []
        new_count = hidden = 0
        for job in jobs:
            key = job_key(job)
            status = (tracker.get(key) or {}).get("status", "")
            if status in HIDDEN_STATUSES:
                hidden += 1
                continue
            entry = seen.get(key)
            new_count += entry is None
            out.append(job.model_copy(update={
                "key": key,
                "is_new": entry is None,
                "first_seen": entry["first_seen"] if entry else now,
                "tracking_status": status,
            }))
        return AnnotateResult(out, new_count, hidden)

    def mark_seen(self, jobs: list[ScoredJob]) -> None:
        """Remember jobs so they aren't flagged as new next time; prune stale entries."""
        with self._lock:
            seen: dict[str, dict] = self._read("seen.json", {})
            now = now_iso()
            for job in jobs:
                key = job.key or job_key(job)
                first = (seen.get(key) or {}).get("first_seen") or job.first_seen or now
                seen[key] = {"first_seen": first, "last_seen": now}
            cutoff = (datetime.now(timezone.utc) - timedelta(days=SEEN_RETENTION_DAYS)).isoformat()
            self._write("seen.json", {k: v for k, v in seen.items() if v.get("last_seen", "") >= cutoff})

    # ------------------------------------------------------------------ searches
    def save_search(
        self,
        response: SearchResponse,
        *,
        location: str,
        queries: list[str],
        trigger: str = "manual",
    ) -> SearchSummary:
        with self._lock:
            search_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
            response = response.model_copy(update={"search_id": search_id})
            summary = SearchSummary(
                id=search_id,
                created_at=now_iso(),
                trigger=trigger,
                location=location,
                queries=queries,
                total_fetched=response.total_fetched,
                total_returned=response.total_returned,
                new_count=response.new_count,
                top_score=max((j.match_score for j in response.jobs), default=0),
                scoring_engine=response.scoring_engine,
            )
            self._write(f"searches/{search_id}.json",
                        StoredSearch(summary=summary, response=response).model_dump())
            index = [summary.model_dump(), *self._read("searches/index.json", [])]
            for old in index[MAX_SEARCHES:]:
                self._path(f"searches/{old['id']}.json").unlink(missing_ok=True)
            self._write("searches/index.json", index[:MAX_SEARCHES])
            return summary

    def list_searches(self, limit: int = MAX_SEARCHES) -> list[SearchSummary]:
        out = []
        for item in self._read("searches/index.json", [])[:limit]:
            try:
                out.append(SearchSummary.model_validate(item))
            except ValidationError:
                continue
        return out

    def load_search(self, search_id: str) -> StoredSearch | None:
        if not search_id.replace("-", "").isalnum():  # no path tricks
            return None
        data = self._read(f"searches/{search_id}.json", None)
        if not data:
            return None
        try:
            stored = StoredSearch.model_validate(data)
        except ValidationError:
            return None
        # Refresh tracking badges: statuses may have changed since the search ran.
        tracker = self._read("tracker.json", {})
        jobs = []
        for j in stored.response.jobs:
            key = j.key or job_key(j)
            status = (tracker.get(key) or {}).get("status", "")
            if status not in HIDDEN_STATUSES:
                jobs.append(j.model_copy(update={"key": key, "tracking_status": status}))
        stored.response.jobs = jobs
        return stored

    def latest_search(self) -> StoredSearch | None:
        searches = self.list_searches(limit=1)
        return self.load_search(searches[0].id) if searches else None

    def combine_searches(self, days: int, now: datetime | None = None) -> CombinedSearch:
        """Merge every saved search from the last `days` calendar days (IST) into one list.

        The same job (same key) across searches appears once: its most recent version is
        kept, with `appearances`, the earliest `first_seen`, and every apply link."""
        now = now or datetime.now(timezone.utc)
        first_day = now.astimezone(IST).date() - timedelta(days=days - 1)
        start = datetime.combine(first_day, dtime(0, 0), tzinfo=IST)
        summaries = [s for s in self.list_searches() if datetime.fromisoformat(s.created_at) >= start]

        merged: dict[str, ScoredJob] = {}
        total = 0
        engines: set[str] = set()
        for summary in reversed(summaries):  # oldest first, so newer versions replace older ones
            stored = self.load_search(summary.id)  # refreshes statuses, hides dismissed
            if stored is None:
                continue
            engines.add(stored.response.scoring_engine)
            for job in stored.response.jobs:
                total += 1
                key = job.key or job_key(job)
                prev = merged.get(key)
                if prev is None:
                    merged[key] = job.model_copy(update={"key": key, "appearances": 1})
                    continue
                links = {job.url: None}
                alt = []
                for link in [*job.alt_links, JobLink(source=prev.source, url=prev.url), *prev.alt_links]:
                    if link.url and link.url not in links:
                        links[link.url] = None
                        alt.append(link)
                merged[key] = job.model_copy(update={
                    "key": key,
                    "appearances": prev.appearances + 1,
                    "first_seen": min(filter(None, [prev.first_seen, job.first_seen]), default=""),
                    "is_new": prev.is_new or job.is_new,
                    "alt_links": alt,
                })

        jobs = sorted(merged.values(), key=lambda j: (j.match_score, j.first_seen), reverse=True)
        response = SearchResponse(
            total_fetched=total,
            total_returned=len(jobs),
            scoring_engine=engines.pop() if len(engines) == 1 else ("mixed" if engines else "local"),
            sources=[],
            duplicates_merged=total - len(jobs),
            jobs=jobs,
            new_count=sum(j.is_new for j in jobs),
        )
        return CombinedSearch(days=days, start=start.isoformat(), end=now.isoformat(timespec="seconds"),
                              searches=summaries, total_listings=total,
                              duplicates_removed=total - len(jobs), response=response)

    # ------------------------------------------------------------------ tracker
    def _tracker(self) -> dict[str, dict]:
        return self._read("tracker.json", {})

    def track(self, job: ScoredJob, status: TrackingStatus = "saved", notes: str = "") -> TrackedJob:
        """Add a job to the tracker, or update it if already tracked."""
        with self._lock:
            tracker = self._tracker()
            key = job.key or job_key(job)
            now = now_iso()
            snapshot = job.model_copy(update={"key": key, "tracking_status": status, "is_new": False})
            if key in tracker:
                item = TrackedJob.model_validate(tracker[key])
                item.job = snapshot  # newest details (salary, links...) win
                if status != item.status:
                    item.history.append(StatusChange(status=status, at=now))
                    item.status = status
                if notes:
                    item.notes = notes
                item.updated_at = now
            else:
                item = TrackedJob(key=key, job=snapshot, status=status, notes=notes, created_at=now,
                                  updated_at=now, history=[StatusChange(status=status, at=now)])
            tracker[key] = item.model_dump()
            self._write("tracker.json", tracker)
            return item

    def update_tracked(self, key: str, status: TrackingStatus | None = None, notes: str | None = None) -> TrackedJob:
        with self._lock:
            tracker = self._tracker()
            if key not in tracker:
                raise KeyError(key)
            item = TrackedJob.model_validate(tracker[key])
            now = now_iso()
            if status and status != item.status:
                item.status = status
                item.job.tracking_status = status
                item.history.append(StatusChange(status=status, at=now))
            if notes is not None:
                item.notes = notes
            item.updated_at = now
            tracker[key] = item.model_dump()
            self._write("tracker.json", tracker)
            return item

    def untrack(self, key: str) -> bool:
        with self._lock:
            tracker = self._tracker()
            if tracker.pop(key, None) is None:
                return False
            self._write("tracker.json", tracker)
            return True

    def get_tracked(self, key: str) -> TrackedJob | None:
        raw = self._tracker().get(key)
        return TrackedJob.model_validate(raw) if raw else None

    def list_tracked(self, status: TrackingStatus | None = None) -> list[TrackedJob]:
        items = []
        for raw in self._tracker().values():
            try:
                item = TrackedJob.model_validate(raw)
            except ValidationError:
                continue
            if status is None or item.status == status:
                items.append(item)
        return sorted(items, key=lambda i: i.updated_at, reverse=True)

    def tracker_stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for raw in self._tracker().values():
            s = raw.get("status", "")
            counts[s] = counts.get(s, 0) + 1
        return counts


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    return Storage(get_env_settings().data_dir)
