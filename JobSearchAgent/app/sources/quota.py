"""Quota guard for metered APIs (JSearch / SerpAPI free tiers).

Tracks the provider-reported remaining calls (persisted in data/quota.json) so
the app never burns through a monthly allowance unnoticed:
  * JSearch reports `x-ratelimit-requests-remaining` / `-reset` on every response.
  * SerpAPI exposes a free /account.json endpoint (doesn't count as a search).
Before a source runs, `allow()` caps how many calls it may make; with nothing
left the source is skipped (status "skipped") until the quota resets.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol


class _DocStore(Protocol):
    def load_doc(self, name: str, default): ...
    def save_doc(self, name: str, data) -> None: ...


def _now() -> datetime:
    return datetime.now(timezone.utc)


class QuotaTracker:
    DOC = "quota.json"

    def __init__(self, store: _DocStore, reserve: int = 0):
        self.store = store
        self.reserve = reserve  # calls to always keep spare

    def _state(self) -> dict:
        return self.store.load_doc(self.DOC, {})

    def remaining(self, source: str) -> int | None:
        """Last known remaining calls, or None if unknown / the quota has reset since."""
        entry = self._state().get(source)
        if not entry or entry.get("remaining") is None:
            return None
        reset_at = entry.get("reset_at")
        if reset_at and datetime.fromisoformat(reset_at) <= _now():
            return None
        return int(entry["remaining"])

    def reset_at(self, source: str) -> str:
        return (self._state().get(source) or {}).get("reset_at") or ""

    def allow(self, source: str, wanted: int) -> int:
        left = self.remaining(source)
        if left is None:
            return wanted
        return max(0, min(wanted, left - self.reserve))

    def update(self, source: str, *, remaining: int | None = None, reset_in_seconds: float | None = None,
               reset_at: str | None = None, used: int = 0) -> None:
        state = self._state()
        entry = state.get(source) or {}
        if remaining is not None:
            entry["remaining"] = int(remaining)
        elif used and entry.get("remaining") is not None:
            entry["remaining"] = max(0, int(entry["remaining"]) - used)
        if reset_in_seconds is not None:
            entry["reset_at"] = (_now() + timedelta(seconds=float(reset_in_seconds))).isoformat(timespec="seconds")
        elif reset_at:
            entry["reset_at"] = reset_at
        entry["checked_at"] = _now().isoformat(timespec="seconds")
        state[source] = entry
        self.store.save_doc(self.DOC, state)

    def skip_message(self, source: str) -> str:
        reset = self.reset_at(source)
        when = f"; resets {datetime.fromisoformat(reset):%d %b}" if reset else ""
        return f"Free API quota used up{when}"

    def snapshot(self) -> dict:
        return self._state()
