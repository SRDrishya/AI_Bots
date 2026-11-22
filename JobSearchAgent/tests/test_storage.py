from __future__ import annotations

import json
import threading

import pytest

from app.models import JobListing, ResumeProfile, ScoredJob, SearchRequest, SearchResponse, SourceStatus
from app.services import search as search_mod
from app.services.scoring import ScoringResult
from app.services.storage import MAX_SEARCHES, Storage, job_key


@pytest.fixture
def store(tmp_path) -> Storage:
    return Storage(tmp_path / "data")


_n = 0


def sjob(title="Data Engineer", company="Acme", score=80, source="jsearch", **kw) -> ScoredJob:
    global _n
    _n += 1
    return ScoredJob(id=f"{source}:{_n}", title=title, company=company, url=f"https://x/{_n}",
                     source=source, match_score=score, **kw)


# ------------------------------------------------------------------ basics
def test_profile_roundtrip(store):
    assert store.load_profile() is None
    profile = ResumeProfile(candidate_name="Priya", skills=["Python"])
    store.save_profile(profile)
    assert store.load_profile() == profile


def test_corrupt_file_is_moved_aside(store):
    store.save_profile(ResumeProfile(candidate_name="Priya"))
    path = store.root / "profile.json"
    path.write_text("{not json", encoding="utf-8")
    assert store.load_profile() is None
    assert any(p.name.startswith("profile.json.corrupt-") for p in store.root.iterdir())
    store.save_profile(ResumeProfile(candidate_name="Again"))  # still usable
    assert store.load_profile().candidate_name == "Again"


def test_writes_are_atomic_and_leave_no_temp_files(store):
    store.save_profile(ResumeProfile())
    assert sorted(p.name for p in store.root.iterdir()) == ["profile.json"]


def test_concurrent_tracking_loses_nothing(store):
    jobs = [sjob(title=f"Role {i}", company=f"Co {i}") for i in range(40)]
    threads = [threading.Thread(target=store.track, args=(j,)) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.list_tracked()) == 40


# ------------------------------------------------------------------ identity
def test_job_key_is_stable_across_sources():
    a = sjob("Senior Data Engineer – Databricks", "5100 Kyndryl Solutions Private Limited", source="jsearch")
    b = sjob("Senior Data Engineer - Databricks", "Kyndryl", source="serpapi")
    c = sjob("Data Scientist", "Kyndryl")
    assert job_key(a) == job_key(b) != job_key(c)


# ------------------------------------------------------------------ new / seen
def test_annotate_and_mark_seen(store):
    a, b = sjob("Data Engineer", "A"), sjob("Data Engineer", "B")
    first = store.annotate([a, b])
    assert first.new_count == 2 and all(j.is_new and j.key for j in first.jobs)
    store.mark_seen(first.jobs[:1])  # only A was shown

    again = store.annotate([sjob("Data Engineer", "A", source="serpapi"), b])
    by_company = {j.company: j for j in again.jobs}
    assert not by_company["A"].is_new  # same job via another source
    assert by_company["A"].first_seen == first.jobs[0].first_seen
    assert by_company["B"].is_new  # never shown -> still new


def test_annotate_does_not_write(store):
    store.annotate([sjob()])
    assert not (store.root / "seen.json").exists()


# ------------------------------------------------------------------ tracker
def test_tracking_lifecycle(store):
    job = sjob("Data Engineer", "Zoho")
    item = store.track(job, "saved", notes="referral from Ravi")
    assert item.status == "saved" and item.key == job_key(job)

    item = store.update_tracked(item.key, status="applied")
    item = store.update_tracked(item.key, status="interviewing", notes="Round 1 on Monday")
    assert [h.status for h in item.history] == ["saved", "applied", "interviewing"]
    assert item.notes == "Round 1 on Monday" and item.job.tracking_status == "interviewing"

    item = store.update_tracked(item.key, status="interviewing")  # no-op status change
    assert len(item.history) == 3

    assert store.tracker_stats() == {"interviewing": 1}
    assert [t.key for t in store.list_tracked("interviewing")] == [item.key]
    assert store.list_tracked("applied") == []

    # Re-tracking the same job from another source updates, not duplicates.
    store.track(sjob("Data Engineer", "Zoho Corporation", source="serpapi", salary="INR 20L"), "offer")
    assert len(store.list_tracked()) == 1
    assert store.get_tracked(item.key).job.salary == "INR 20L"

    assert store.untrack(item.key) and not store.untrack(item.key)
    with pytest.raises(KeyError):
        store.update_tracked(item.key, status="saved")


def test_annotate_shows_status_and_hides_dismissed(store):
    applied, dismissed, plain = sjob("Data Engineer", "A"), sjob("Data Engineer", "B"), sjob("Data Engineer", "C")
    store.track(applied, "applied")
    store.track(dismissed, "dismissed")
    result = store.annotate([applied, dismissed, plain])
    assert result.hidden_count == 1
    assert {j.company: j.tracking_status for j in result.jobs} == {"A": "applied", "C": ""}


# ------------------------------------------------------------------ searches
def _response(jobs) -> SearchResponse:
    return SearchResponse(total_fetched=len(jobs), total_returned=len(jobs), scoring_engine="local",
                          sources=[], jobs=jobs, new_count=1)


def test_search_history_roundtrip_and_pruning(store):
    first = store.save_search(_response([sjob(score=91)]), location="Chennai", queries=["Data Engineer"])
    assert first.top_score == 91 and first.trigger == "manual"
    loaded = store.load_search(first.id)
    assert loaded.response.search_id == first.id and loaded.response.jobs[0].match_score == 91

    for _ in range(MAX_SEARCHES + 3):
        store.save_search(_response([]), location="Chennai", queries=["x"], trigger="scheduled")
    summaries = store.list_searches()
    assert len(summaries) == MAX_SEARCHES
    assert store.load_search(first.id) is None  # pruned, file deleted
    assert len(list((store.root / "searches").glob("*.json"))) == MAX_SEARCHES + 1  # + index.json
    assert store.latest_search().summary.id == summaries[0].id


def test_loaded_search_reflects_current_tracking(store):
    job = store.annotate([sjob("Data Engineer", "A"), sjob("Data Engineer", "B")]).jobs
    summary = store.save_search(_response(job), location="Chennai", queries=["x"])
    store.track(job[0], "applied")
    store.track(job[1], "dismissed")
    loaded = store.load_search(summary.id)
    assert [(j.company, j.tracking_status) for j in loaded.response.jobs] == [("A", "applied")]


def test_load_search_rejects_path_tricks(store):
    assert store.load_search("../profile") is None


# ------------------------------------------------------------------ pipeline
async def test_run_search_pipeline(store, monkeypatch):
    raw = [
        JobListing(id="jsearch:1", title="Data Engineer", company="A", location="Chennai", url="u1", source="jsearch"),
        JobListing(id="serpapi:1", title="Data Engineer", company="A Pvt Ltd", location="Chennai", url="u2", source="serpapi"),
        JobListing(id="jsearch:2", title="Data Engineer", company="B", location="Remote (USA)", remote=True, url="u3", source="jsearch"),
        JobListing(id="jsearch:3", title="Data Engineer", company="C", location="Remote (India)", remote=True, url="u4", source="jsearch"),
        JobListing(id="jsearch:4", title="Data Engineer", company="D", location="Chennai", url="u5", source="jsearch"),
        *[JobListing(id=f"jsearch:x{c}", title="Data Engineer", company=c, location="Chennai", url=f"ux{c}",
                     source="jsearch") for c in "EFGH"],
    ]

    async def fake_fetch_all(request, ctx=None):
        return raw, [SourceStatus(source="jsearch", status="ok", count=4)]

    seen_profiles = []

    async def fake_score(jobs, profile, key, model):
        seen_profiles.append((key, model))
        scored = [ScoredJob(**j.model_dump(), match_score=90 - i) for i, j in enumerate(jobs)]
        return ScoringResult(scored, "local")

    monkeypatch.setattr(search_mod, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(search_mod, "score_jobs", fake_score)

    request = SearchRequest(profile=ResumeProfile(search_queries=["Data Engineer Chennai"]), max_results=5)
    store.track(ScoredJob(**raw[4].model_dump(), match_score=1), "dismissed")

    resp = await search_mod.run_search(request, storage=store)
    assert resp.total_fetched == 9 and resp.total_returned == 5 and resp.new_count == 5
    assert resp.excluded == {"remote_restricted": 1, "dismissed": 1}
    assert resp.duplicates_merged == 1
    assert [j.company for j in resp.jobs] == ["A", "C", "E", "F", "G"]  # D dismissed, H cut
    assert resp.search_id and store.load_search(resp.search_id).summary.queries == ["Data Engineer"]

    again = await search_mod.run_search(request, storage=store, trigger="scheduled")
    assert again.new_count == 0
    assert again.total_returned == 5
    assert store.list_searches()[0].trigger == "scheduled"


async def test_run_search_requires_terms(store):
    with pytest.raises(ValueError):
        await search_mod.run_search(SearchRequest(profile=ResumeProfile()), storage=store)


# ------------------------------------------------------------------ resume file
@pytest.mark.parametrize(
    ("given", "stored"),
    [
        ("Priya Resume 2026.PDF", "Priya Resume 2026.pdf"),
        ("../../etc/passwd.txt", "passwd.txt"),
        (r"C:\Users\x\cv.docx", "cv.docx"),
        ("weird<>:|?*name.docx", "weird_name.docx"),
        ("", "resume"),
    ],
)
def test_resume_file_names_are_safe(store, given, stored):
    info = store.save_resume_file(given, b"data")
    assert info.filename == stored
    path = store.resume_file_path()
    assert path.parent == store.root / "resume" and path.read_bytes() == b"data"


def test_resume_file_replaced_and_info_tracks_disk(store):
    assert store.resume_file_info() is None
    first = store.save_resume_file("a.pdf", b"one")
    second = store.save_resume_file("b.docx", b"two")
    assert first.sha256 != second.sha256
    assert [p.name for p in (store.root / "resume").iterdir()] == ["b.docx"]
    (store.root / "resume" / "b.docx").unlink()  # user deleted the file by hand
    assert store.resume_file_info() is None


# ------------------------------------------------------------------ combined view (last N days)
def _save_at(store, monkeypatch, iso_utc, jobs):
    monkeypatch.setattr("app.services.storage.now_iso", lambda: iso_utc)
    return store.save_search(_response(jobs), location="Chennai", queries=["Data Engineer"], trigger="scheduled")


def _sj(company, score, url, first_seen, is_new=False):
    return ScoredJob(id=f"jsearch:{company}:{url}", title="Data Engineer", company=company, url=url,
                     source="jsearch", match_score=score, first_seen=first_seen, is_new=is_new)


def test_combine_last_three_days(store, monkeypatch):
    from datetime import datetime

    # 08:00 IST == 02:30 UTC
    _save_at(store, monkeypatch, "2026-11-18T02:30:00+00:00", [_sj("Old Co", 99, "https://old/1", "2026-11-18T02:30:00+00:00", True)])
    _save_at(store, monkeypatch, "2026-11-21T02:30:00+00:00", [
        _sj("Acme", 70, "https://linkedin/acme", "2026-11-21T02:30:00+00:00", True),
        _sj("Beta", 60, "https://beta/1", "2026-11-19T02:30:00+00:00")])
    _save_at(store, monkeypatch, "2026-11-22T02:30:00+00:00", [
        _sj("Acme", 80, "https://naukri/acme", "2026-11-21T02:30:00+00:00"),
        _sj("Gamma", 65, "https://gamma/1", "2026-11-22T02:30:00+00:00", True)])
    _save_at(store, monkeypatch, "2026-11-23T02:30:00+00:00", [
        _sj("Acme Pvt Ltd", 75, "https://acme.com/jobs/1", "2026-11-21T02:30:00+00:00"),
        _sj("Gamma", 66, "https://gamma/1", "2026-11-22T02:30:00+00:00"),
        _sj("Delta", 90, "https://delta/1", "2026-11-23T02:30:00+00:00", True)])
    store.track(_sj("Delta", 90, "https://delta/1", ""), "dismissed")
    store.track(_sj("Beta", 60, "https://beta/1", ""), "applied")

    now = datetime.fromisoformat("2026-11-23T06:30:00+00:00")  # 12:00 IST
    c = store.combine_searches(3, now=now)
    assert c.start == "2026-11-21T00:00:00+05:30" and len(c.searches) == 3
    assert c.total_listings == 6 and c.duplicates_removed == 3  # Delta hidden (dismissed)
    by = {j.company: j for j in c.response.jobs}
    assert set(by) == {"Acme Pvt Ltd", "Gamma", "Beta"}  # Old Co is outside the window

    acme = by["Acme Pvt Ltd"]  # newest version wins
    assert acme.match_score == 75 and acme.url == "https://acme.com/jobs/1" and acme.appearances == 3
    assert {l.url for l in acme.alt_links} == {"https://naukri/acme", "https://linkedin/acme"}
    assert acme.first_seen == "2026-11-21T02:30:00+00:00" and acme.is_new  # new on day 1 of window
    assert by["Gamma"].appearances == 2 and by["Gamma"].alt_links == []  # same URL not repeated
    assert by["Beta"].tracking_status == "applied" and not by["Beta"].is_new
    assert [j.company for j in c.response.jobs] == ["Acme Pvt Ltd", "Gamma", "Beta"]  # by score
    assert c.response.new_count == 2

    today = store.combine_searches(1, now=now)
    assert [s.created_at[:10] for s in today.searches] == ["2026-11-23"]
    assert [j.company for j in today.response.jobs] == ["Acme Pvt Ltd", "Gamma"]

    # Day boundaries are India time: 23 Nov 00:10 IST is still "today" at 12:00 IST.
    _save_at(store, monkeypatch, "2026-11-22T18:40:00+00:00", [_sj("Late", 50, "https://late/1", "")])
    assert "Late" in {j.company for j in store.combine_searches(1, now=now).response.jobs}


def test_combine_with_no_searches(store):
    c = store.combine_searches(7)
    assert c.searches == [] and c.response.jobs == [] and c.duplicates_removed == 0
