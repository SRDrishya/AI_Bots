from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from app.models import ResumeProfile, SearchResponse
from app.services import scheduler as sched_mod
from app.services.scheduler import IST, ScheduleSettings, SearchScheduler, most_recent_slot, next_slot
from app.services.storage import Storage


class Clock:
    def __init__(self, iso: str):
        self.now = datetime.fromisoformat(iso)

    def __call__(self) -> datetime:
        return self.now

    def set(self, iso: str) -> None:
        self.now = datetime.fromisoformat(iso)


@pytest.fixture
def store(tmp_path) -> Storage:
    s = Storage(tmp_path / "data")
    s.save_profile(ResumeProfile(search_queries=["Data Engineer"]))
    return s


def make(store, clock, calls, *, fail=False, delay=0.0):
    async def fake_run(request, storage, trigger):
        calls.append((request.location, request.max_results, trigger))
        if delay:
            await asyncio.sleep(delay)
        if fail:
            raise RuntimeError("network down")
        return SearchResponse(total_fetched=10, total_returned=7, scoring_engine="local", sources=[],
                              jobs=[], new_count=3, search_id="s1")

    return SearchScheduler(store, run=fake_run, clock=clock)


# ------------------------------------------------------------------ slots (IST)
def test_slots_are_in_india_time():
    # 02:00 UTC == 07:30 IST -> most recent 08:00 IST slot was yesterday.
    now = datetime(2026, 11, 22, 2, 0, tzinfo=timezone.utc)
    assert most_recent_slot(now, "08:00") == datetime(2026, 11, 21, 8, 0, tzinfo=IST)
    assert next_slot(now, "08:00") == datetime(2026, 11, 22, 8, 0, tzinfo=IST)
    # 03:00 UTC == 08:30 IST -> today's slot has passed.
    now = datetime(2026, 11, 22, 3, 0, tzinfo=timezone.utc)
    assert most_recent_slot(now, "08:00") == datetime(2026, 11, 22, 8, 0, tzinfo=IST)


@pytest.mark.parametrize(("raw", "ok"), [("8:05", "08:05"), ("23:59", "23:59"), (" 07:00 ", "07:00")])
def test_run_time_normalised(raw, ok):
    assert ScheduleSettings(run_time=raw).run_time == ok


@pytest.mark.parametrize("bad", ["24:00", "8", "08:60", "eight"])
def test_run_time_rejected(bad):
    with pytest.raises(ValueError):
        ScheduleSettings(run_time=bad)


# ------------------------------------------------------------------ due logic
async def test_due_lifecycle(store):
    clock, calls = Clock("2026-11-22T03:00:00+00:00"), []   # 08:30 IST
    s = make(store, clock, calls)
    assert s.is_due()  # never ran -> catch up

    state = await s.run_now()
    assert state.last_status == "ok" and state.last_new_count == 3 and state.last_search_id == "s1"
    assert calls == [("Chennai", 50, "scheduled")]
    assert not s.is_due()  # already ran for today's slot

    clock.set("2026-11-23T02:29:00+00:00")  # 07:59 IST next day
    assert not s.is_due()
    clock.set("2026-11-23T02:30:00+00:00")  # 08:00 IST
    assert s.is_due()

    clock.set("2026-11-26T10:00:00+00:00")  # app closed for 3 days -> one catch-up, not three
    await s.run_now()
    assert not s.is_due()


async def test_disabled_and_settings(store):
    clock, calls = Clock("2026-11-22T03:00:00+00:00"), []
    s = make(store, clock, calls)
    s.update_settings(enabled=False)
    assert not s.is_due() and s.status().next_run_at == ""
    s.update_settings(enabled=True, run_time="21:15", remote_only=True, max_results=30)
    status = s.status()
    assert status.next_run_at == "2026-11-22T21:15+05:30"
    await s.run_now()
    assert calls == [("Chennai", 30, "scheduled")]
    with pytest.raises(ValueError):
        s.update_settings(run_time="99:99")
    assert s.settings().run_time == "21:15"  # invalid update not saved


async def test_no_profile_is_skipped(tmp_path):
    clock, calls = Clock("2026-11-22T03:00:00+00:00"), []
    empty = Storage(tmp_path / "empty")
    s = make(empty, clock, calls)
    assert not s.is_due()  # nothing to search for yet
    state = await s.run_now()  # manual "Run now" explains why
    assert state.last_status == "skipped" and "No resume profile" in state.last_message
    assert calls == []
    empty.save_profile(ResumeProfile(search_queries=["Data Engineer"]))
    assert s.is_due()  # resume uploaded -> today's search catches up immediately


async def test_failure_recorded_without_retry_storm(store):
    clock, calls = Clock("2026-11-22T03:00:00+00:00"), []
    s = make(store, clock, calls, fail=True)
    state = await s.run_now()
    assert state.last_status == "error" and state.last_message == "network down"
    assert not s.is_due()  # next attempt is tomorrow's slot


async def test_concurrent_run_now_runs_once(store):
    clock, calls = Clock("2026-11-22T03:00:00+00:00"), []
    s = make(store, clock, calls, delay=0.05)
    first, second = await asyncio.gather(s.run_now(), s.run_now("run_now"))
    assert len(calls) == 1
    assert "already running" in second.last_message


async def test_background_loop_runs_when_due_and_stops(store, monkeypatch):
    monkeypatch.setattr(sched_mod, "STARTUP_DELAY", 0)
    monkeypatch.setattr(sched_mod, "TICK_SECONDS", 0.01)
    clock, calls = Clock("2026-11-22T03:00:00+00:00"), []
    s = make(store, clock, calls)
    s.start()
    await asyncio.sleep(0.1)
    await s.stop()
    assert len(calls) == 1  # ran once for the slot, then idle
    assert s.status().state.last_status == "ok"
