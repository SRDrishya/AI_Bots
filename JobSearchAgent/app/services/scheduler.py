"""Daily scheduled search (in-process, no extra dependencies).

One rule covers both the normal daily run and catching up after the app was
closed / the laptop slept:

    due  <=>  enabled  and  a profile is saved  and
              no attempt since the most recent scheduled slot (HH:MM IST)

A background task checks this every TICK_SECONDS. Settings and run state are
persisted in data/schedule.json. India has no DST, so a fixed +05:30 offset is
exact and needs no tz database.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import datetime, time, timedelta, timezone

from pydantic import BaseModel, Field, field_validator

from app.config import IST
from app.models import SearchRequest, SearchResponse
from app.services.search import run_search
from app.services.storage import Storage

log = logging.getLogger(__name__)

TICK_SECONDS = 30.0
STARTUP_DELAY = 20.0  # let the server finish booting before a catch-up run


class ScheduleSettings(BaseModel):
    enabled: bool = True
    run_time: str = Field(default="08:00", description="HH:MM, India time")
    location: str = "Chennai"
    remote_only: bool = False
    max_results: int = Field(default=50, ge=5, le=200)

    @field_validator("run_time")
    @classmethod
    def _valid_time(cls, v: str) -> str:
        m = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", v.strip())
        if not m:
            raise ValueError("run_time must be HH:MM (24h)")
        return f"{int(m.group(1)):02d}:{m.group(2)}"


class ScheduleState(BaseModel):
    last_attempt_at: str = ""
    last_status: str = ""          # ok | error | skipped
    last_message: str = ""
    last_search_id: str = ""
    last_new_count: int = 0
    last_total: int = 0
    last_email_message: str = ""   # result of the email scan that runs with the daily search


class ScheduleStatus(BaseModel):
    settings: ScheduleSettings
    state: ScheduleState
    running: bool
    next_run_at: str


def _slot_time(run_time: str) -> time:
    h, m = run_time.split(":")
    return time(int(h), int(m), tzinfo=IST)


def most_recent_slot(now: datetime, run_time: str) -> datetime:
    local = now.astimezone(IST)
    slot = datetime.combine(local.date(), _slot_time(run_time))
    return slot if slot <= local else slot - timedelta(days=1)


def next_slot(now: datetime, run_time: str) -> datetime:
    return most_recent_slot(now, run_time) + timedelta(days=1)


RunFn = Callable[..., Awaitable[SearchResponse]]


class SearchScheduler:
    DOC = "schedule.json"

    def __init__(self, storage: Storage, run: RunFn = run_search,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 email_scan: Callable[[], Awaitable[dict]] | None = None):
        self.storage = storage
        self._email_scan = email_scan
        self._run = run
        self._clock = clock
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------ persistence
    def _doc(self) -> dict:
        return self.storage.load_doc(self.DOC, {})

    def settings(self) -> ScheduleSettings:
        try:
            return ScheduleSettings.model_validate(self._doc().get("settings", {}))
        except ValueError:
            return ScheduleSettings()

    def state(self) -> ScheduleState:
        try:
            return ScheduleState.model_validate(self._doc().get("state", {}))
        except ValueError:
            return ScheduleState()

    def _save(self, settings: ScheduleSettings | None = None, state: ScheduleState | None = None) -> None:
        doc = self._doc()
        if settings is not None:
            doc["settings"] = settings.model_dump()
        if state is not None:
            doc["state"] = state.model_dump()
        self.storage.save_doc(self.DOC, doc)

    def update_settings(self, **changes) -> ScheduleSettings:
        merged = ScheduleSettings.model_validate({**self.settings().model_dump(), **changes})
        self._save(settings=merged)
        return merged

    def status(self) -> ScheduleStatus:
        s = self.settings()
        return ScheduleStatus(
            settings=s,
            state=self.state(),
            running=self._lock.locked(),
            next_run_at=next_slot(self._clock(), s.run_time).isoformat(timespec="minutes") if s.enabled else "",
        )

    # ------------------------------------------------------------------ decision
    def is_due(self) -> bool:
        s = self.settings()
        if not s.enabled or self._lock.locked():
            return False
        if self.storage.load_profile() is None:  # wait for a resume; then catch up right away
            return False
        last = self.state().last_attempt_at
        slot = most_recent_slot(self._clock(), s.run_time)
        return not last or datetime.fromisoformat(last) < slot

    # ------------------------------------------------------------------ running
    async def run_now(self, trigger: str = "scheduled") -> ScheduleState:
        """Run the scheduled search immediately (also used by the 'Run now' button)."""
        if self._lock.locked():
            return self.state().model_copy(update={"last_message": "A scheduled search is already running"})
        async with self._lock:
            settings = self.settings()
            profile = self.storage.load_profile()
            if profile is None:  # not an attempt: today's search still runs once a resume exists
                state = self.state().model_copy(update={"last_status": "skipped",
                                                        "last_message": "No resume profile saved yet"})
                self._save(state=state)
                return state
            state = ScheduleState(last_attempt_at=self._clock().isoformat(timespec="seconds"))
            self._save(state=state)  # record the attempt first: a crash mustn't cause a retry storm
            request = SearchRequest(profile=profile, location=settings.location,
                                    remote_only=settings.remote_only, max_results=settings.max_results)
            try:
                response = await self._run(request, storage=self.storage, trigger="scheduled")
            except Exception as exc:
                log.exception("Scheduled search failed")
                state.last_status, state.last_message = "error", str(exc)[:300] or type(exc).__name__
            else:
                state.last_status = "ok"
                state.last_search_id = response.search_id
                state.last_new_count = response.new_count
                state.last_total = response.total_returned
                state.last_message = (f"{response.new_count} new of {response.total_returned} matches"
                                      + (f" ({trigger})" if trigger != "scheduled" else "")
                                      + (f" · {response.email_notice}" if response.email_notice else ""))
                log.info("Scheduled search: %s", state.last_message)
            state.last_email_message = await self._scan_emails()
            self._save(state=state)
            return state

    async def _scan_emails(self) -> str:
        """Pick up job emails received since the previous scan; never fails the daily run."""
        if self._email_scan is None or not any(self.storage.load_doc("email_accounts.json", {}).values()):
            return ""
        try:
            result = await self._email_scan()
        except Exception as exc:
            log.exception("Scheduled email scan failed")
            return f"Email scan failed: {str(exc)[:200] or type(exc).__name__}"
        message = f"{result.get('new_count', 0)} new job emails"
        if result.get("errors"):
            message += f" ({len(result['errors'])} inbox(es) failed — reconnect in Email jobs)"
        log.info("Scheduled email scan: %s", message)
        return message

    async def _loop(self) -> None:
        await asyncio.sleep(STARTUP_DELAY)
        while True:
            try:
                if self.is_due():
                    await self.run_now()
            except Exception:  # never let the loop die
                log.exception("Scheduler tick failed")
            await asyncio.sleep(TICK_SECONDS)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="search-scheduler")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
