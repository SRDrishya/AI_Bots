from __future__ import annotations

import pytest

from app.config import get_env_settings
from app.models import ScoredJob, SearchResponse
from app.services import notify


def job(i: int, *, new: bool, score: int = 60) -> ScoredJob:
    return ScoredJob(id=str(i), title=f"Engineer <{i}>", company=f"Co{i}", location="Chennai",
                     url=f"https://jobs.example/{i}?a=1&b=2", source="adzuna", match_score=score, is_new=new)


def response(jobs: list[ScoredJob]) -> SearchResponse:
    return SearchResponse(total_fetched=len(jobs), total_returned=len(jobs), scoring_engine="local",
                          sources=[], jobs=jobs, new_count=sum(j.is_new for j in jobs), search_id="s1")


@pytest.fixture
def smtp_env(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "me@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "abcd efgh")
    monkeypatch.setenv("NOTIFY_EMAIL_TO", "a@example.com, b@example.com")
    return get_env_settings()


@pytest.fixture
def sent(monkeypatch):
    out = []
    monkeypatch.setattr(notify, "_send", lambda msg, env: out.append(msg))
    return out


def test_env_parsing(smtp_env):
    assert smtp_env.smtp_port == 587
    assert smtp_env.smtp_password == "abcdefgh"
    assert smtp_env.notify_email_to == ["a@example.com", "b@example.com"]
    assert notify.is_configured(smtp_env)


def test_message_lists_only_new_jobs_best_first(smtp_env):
    resp = response([job(1, new=False, score=95), job(2, new=True, score=55), job(3, new=True, score=80)])
    msg = notify.build_message(resp, smtp_env, trigger="scheduled", location="Chennai")
    assert msg["Subject"] == "JobPilot: 2 new jobs in Chennai"
    assert msg["To"] == "a@example.com, b@example.com" and msg["From"] == "me@example.com"
    text = msg.get_body(("plain",)).get_content()
    assert "Engineer <1>" not in text
    assert text.index("Engineer <3>") < text.index("Engineer <2>")
    html = msg.get_body(("html",)).get_content()
    assert "Engineer &lt;3&gt;" in html and "?a=1&amp;b=2" in html


async def test_not_configured_does_nothing(sent):
    assert await notify.notify_new_jobs(response([job(1, new=True)])) == ""
    assert sent == []


async def test_sends_when_new_jobs(smtp_env, sent):
    msg = await notify.notify_new_jobs(response([job(1, new=True), job(2, new=False)]), env=smtp_env)
    assert msg == "Emailed 1 new jobs" and len(sent) == 1


async def test_skips_when_nothing_new_unless_asked(smtp_env, sent, monkeypatch):
    assert "not sent" in await notify.notify_new_jobs(response([job(1, new=False)]), env=smtp_env)
    assert sent == []
    monkeypatch.setenv("NOTIFY_SEND_EMPTY", "true")
    await notify.notify_new_jobs(response([job(1, new=False)]))
    assert sent[0]["Subject"] == "JobPilot: no new jobs"


async def test_send_failure_is_reported_not_raised(smtp_env, monkeypatch):
    def boom(msg, env):
        raise OSError("connection refused")
    monkeypatch.setattr(notify, "_send", boom)
    assert await notify.notify_new_jobs(response([job(1, new=True)]), env=smtp_env) == \
        "Email failed: connection refused"
