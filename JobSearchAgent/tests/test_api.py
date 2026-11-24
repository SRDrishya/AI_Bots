from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app import main as main_mod
from app.models import ScoredJob, SearchResponse, SourceStatus
from app.services.storage import Storage
from tests.test_resume import SAMPLE


@pytest.fixture
def store(tmp_path) -> Storage:
    return Storage(tmp_path / "data")


@pytest.fixture
def client(store, monkeypatch):
    async def fake_run_search(request, storage, trigger):
        jobs = [ScoredJob(id="jsearch:1", title="Data Engineer", company="Zoho", location="Chennai",
                          url="https://zoho.com/jobs/1", source="jsearch", match_score=88, location_match="local")]
        annotated = storage.annotate(jobs)
        storage.mark_seen(annotated.jobs)
        resp = SearchResponse(total_fetched=5, total_returned=1, scoring_engine="local",
                              sources=[SourceStatus(source="jsearch", status="ok", count=5)],
                              jobs=annotated.jobs, new_count=annotated.new_count)
        summary = storage.save_search(resp, location=request.location, queries=["Data Engineer"], trigger=trigger)
        return resp.model_copy(update={"search_id": summary.id})

    monkeypatch.setattr(main_mod, "run_search", fake_run_search)
    monkeypatch.setattr("app.services.scheduler.run_search", fake_run_search)
    app = main_mod.create_app(store, start_scheduler=False)
    # The scheduler captured run_search at construction; point it at the fake too.
    app.state.scheduler._run = fake_run_search
    with TestClient(app) as c:
        yield c


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "JobPilot AI" in r.text


def test_status_without_keys(client):
    s = client.get("/api/status").json()
    assert s["has_profile"] is False and s["gemini_configured"] is False
    by_name = {x["name"]: x for x in s["sources"]}
    assert by_name["himalayas"] == {"name": "himalayas", "needs_key": False, "configured": True}
    assert by_name["jsearch"]["needs_key"] and not by_name["jsearch"]["configured"]
    assert s["schedule"]["settings"]["run_time"] == "08:00"


def test_email_routes_are_read_only_and_require_oauth_configuration(client, monkeypatch):
    monkeypatch.setattr("app.services.email_jobs.keyring.get_password", lambda *args: None)

    accounts = client.get("/api/emails/accounts")
    assert accounts.status_code == 200
    assert [a["provider"] for a in accounts.json()["accounts"]] == ["google", "microsoft"]
    assert all(not a["configured"] and not a["connected"] for a in accounts.json()["accounts"])
    assert client.get("/api/emails/connect/google").status_code == 400
    saved = client.get("/api/emails/jobs").json()
    assert saved["jobs"] == [] and saved["errors"] == [] and saved["new_count"] == 0
    scan = client.post("/api/emails/scan").json()
    assert scan["jobs"] == [] and scan["scanned_since"] == {}


def test_search_requires_profile(client):
    r = client.post("/api/search", json={})
    assert r.status_code == 400 and "resume" in r.json()["detail"].lower()


def test_resume_upload_and_profile_edit(client):
    r = client.post("/api/resume", files={"file": ("cv.txt", SAMPLE.encode(), "text/plain")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reused"] is False and body["file"]["filename"] == "cv.txt"
    profile = body["profile"]
    assert profile["candidate_name"] == "Priya Ramesh" and profile["analysis_engine"] == "local"
    assert client.get("/api/profile").json() == profile

    profile["target_roles"] = ["Analytics Engineer"]
    assert client.put("/api/profile", json=profile).json()["target_roles"] == ["Analytics Engineer"]
    assert client.get("/api/status").json()["has_profile"] is True


def test_bad_resume_is_400(client):
    r = client.post("/api/resume", files={"file": ("cv.png", b"x" * 100, "image/png")})
    assert r.status_code == 400 and "Unsupported" in r.json()["detail"]


def test_search_history_and_tracker_flow(client):
    client.post("/api/resume", files={"file": ("cv.txt", SAMPLE.encode(), "text/plain")})
    assert client.get("/api/searches/latest").status_code == 404

    resp = client.post("/api/search", json={"location": "Chennai", "max_results": 20}).json()
    assert resp["total_returned"] == 1 and resp["new_count"] == 1 and resp["search_id"]
    job = resp["jobs"][0]

    history = client.get("/api/searches").json()
    assert [h["id"] for h in history] == [resp["search_id"]]
    assert client.get("/api/searches/latest").json()["summary"]["id"] == resp["search_id"]
    assert client.get("/api/searches/nope").status_code == 404

    tracked = client.post("/api/tracker", json={"job": job, "status": "applied"}).json()
    key = tracked["key"]
    assert client.get("/api/tracker").json()[0]["status"] == "applied"
    assert client.get("/api/tracker?status=saved").json() == []

    updated = client.patch(f"/api/tracker/{key}", json={"status": "interviewing", "notes": "HR call Mon"}).json()
    assert updated["status"] == "interviewing" and updated["notes"] == "HR call Mon"
    assert client.get("/api/status").json()["tracker"] == {"interviewing": 1}

    # Re-opened search shows the current tracking status.
    reopened = client.get(f"/api/searches/{resp['search_id']}").json()
    assert reopened["response"]["jobs"][0]["tracking_status"] == "interviewing"

    assert client.patch("/api/tracker/missing", json={"status": "saved"}).status_code == 404
    assert client.delete(f"/api/tracker/{key}").status_code == 204
    assert client.delete(f"/api/tracker/{key}").status_code == 404

    second = client.post("/api/search", json={}).json()
    assert second["new_count"] == 0


def test_search_validation(client):
    client.post("/api/resume", files={"file": ("cv.txt", SAMPLE.encode(), "text/plain")})
    assert client.post("/api/search", json={"max_results": 1}).status_code == 422


def test_schedule_settings_and_run_now(client, store):
    r = client.put("/api/schedule", json={"run_time": "07:30", "location": "Chennai", "enabled": True})
    assert r.json()["settings"]["run_time"] == "07:30"
    assert client.put("/api/schedule", json={"run_time": "25:00"}).status_code in (400, 422)

    # No profile yet -> run is recorded as skipped.
    client.post("/api/schedule/run")
    for _ in range(50):
        state = client.get("/api/schedule").json()
        if not state["running"] and state["state"]["last_status"]:
            break
        asyncio.run(asyncio.sleep(0.02))
    assert state["state"]["last_status"] == "skipped"

    client.post("/api/resume", files={"file": ("cv.txt", SAMPLE.encode(), "text/plain")})
    client.post("/api/schedule/run")
    for _ in range(50):
        state = client.get("/api/schedule").json()
        if not state["running"] and state["state"]["last_status"] == "ok":
            break
        asyncio.run(asyncio.sleep(0.02))
    assert state["state"]["last_status"] == "ok" and state["state"]["last_new_count"] == 1
    assert client.get("/api/searches").json()[0]["trigger"] == "scheduled"


# ------------------------------------------------------------------ export
def test_exports(client):
    import io

    from openpyxl import load_workbook

    client.post("/api/resume", files={"file": ("cv.txt", SAMPLE.encode(), "text/plain")})
    assert client.get("/api/searches/latest/export").status_code == 404
    resp = client.post("/api/search", json={}).json()

    x = client.get(f"/api/searches/{resp['search_id']}/export?format=xlsx")
    assert x.status_code == 200 and "spreadsheetml" in x.headers["content-type"]
    assert 'filename="job-matches-' in x.headers["content-disposition"]
    wb = load_workbook(io.BytesIO(x.content))
    assert wb["Matches"].cell(2, 2).value == "Data Engineer"
    assert {r[0].value: r[1].value for r in wb["Profile & search"].iter_rows()}["Candidate"] == "Priya Ramesh"

    c = client.get("/api/searches/latest/export?format=csv")
    assert c.status_code == 200 and c.headers["content-type"].startswith("text/csv")
    assert "Zoho" in c.content.decode("utf-8-sig")

    assert client.get("/api/searches/latest/export?format=pdf").status_code == 400

    client.post("/api/tracker", json={"job": resp["jobs"][0], "status": "applied"})
    t = client.get("/api/tracker/export?format=csv")
    assert t.status_code == 200 and "applied" in t.content.decode("utf-8-sig")
    assert client.get("/api/tracker/export").status_code == 200  # xlsx default


# ------------------------------------------------------------------ application kit
def test_application_kit(client, monkeypatch):
    from app import main as m
    from app.models import ApplicationKit
    from app.services.gemini import GeminiError

    job = {"id": "jsearch:1", "title": "Data Engineer", "company": "Zoho", "url": "https://z/1",
           "source": "jsearch", "match_score": 80}

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = client.post("/api/application-kit", json={"job": job})
    assert r.status_code == 400 and "Gemini" in r.json()["detail"]

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert client.post("/api/application-kit", json={"job": job}).status_code == 400  # no resume yet
    client.post("/api/resume", files={"file": ("cv.txt", SAMPLE.encode(), "text/plain")})

    calls = []

    async def fake_kit(job, profile, resume_text, *, api_key, model, tone):
        calls.append((job.key, tone, resume_text[:12]))
        if tone == "concise":
            raise GeminiError("Gemini quota / rate limit reached")
        return ApplicationKit(job_key=job.key, tone=tone, cover_letter=f"Letter {len(calls)}",
                              created_at="2026-11-22T10:00:00+00:00")

    monkeypatch.setattr(m, "generate_kit", fake_kit)
    first = client.post("/api/application-kit", json={"job": job}).json()
    assert first["cover_letter"] == "Letter 1" and first["job_key"]
    assert calls[0][2] == "PRIYA RAMESH"  # full resume text saved at upload is used

    cached = client.post("/api/application-kit", json={"job": job}).json()
    assert cached["cover_letter"] == "Letter 1" and len(calls) == 1  # served from cache

    regen = client.post("/api/application-kit", json={"job": job, "regenerate": True}).json()
    assert regen["cover_letter"] == "Letter 2"

    warm = client.post("/api/application-kit", json={"job": job, "tone": "warm"}).json()
    assert warm["tone"] == "warm" and len(calls) == 3  # different tone -> new letter

    err = client.post("/api/application-kit", json={"job": job, "tone": "concise"})
    assert err.status_code == 502 and "quota" in err.json()["detail"]


# ------------------------------------------------------------------ resume kept locally, analysed once
def test_resume_saved_and_not_reanalysed(client, store, monkeypatch):
    from app import main as m

    calls = []
    real_analyze = m.analyze_text

    async def counting_analyze(text, key, model):
        calls.append(text[:12])
        return await real_analyze(text, "", model)

    monkeypatch.setattr(m, "analyze_text", counting_analyze)
    pdf_like = SAMPLE.encode()

    first = client.post("/api/resume", files={"file": ("My CV.txt", pdf_like, "text/plain")}).json()
    assert first["reused"] is False and len(calls) == 1
    assert (store.root / "resume" / "My CV.txt").read_bytes() == pdf_like  # original kept on disk

    # User edits the profile, then uploads the same file again: no AI call, edits survive.
    edited = {**first["profile"], "target_roles": ["Analytics Engineer"]}
    client.put("/api/profile", json=edited)
    again = client.post("/api/resume", files={"file": ("renamed.txt", pdf_like, "text/plain")}).json()
    assert again["reused"] is True and len(calls) == 1
    assert again["profile"]["target_roles"] == ["Analytics Engineer"]

    # Saved file info + download.
    info = client.get("/api/resume").json()
    assert info["filename"] == "My CV.txt" and info["size"] == len(pdf_like)
    download = client.get("/api/resume/file")
    assert download.status_code == 200 and download.content == pdf_like
    assert "My%20CV.txt" in download.headers["content-disposition"]  # RFC 5987 encoding

    # Explicit re-analyse runs the AI again on the saved text (and resets edits).
    redo = client.post("/api/resume/reanalyze").json()
    assert len(calls) == 2 and redo["profile"]["target_roles"] != ["Analytics Engineer"]

    # A different resume is analysed and replaces the old file.
    changed = SAMPLE.replace("Snowflake", "BigQuery").encode()
    new = client.post("/api/resume", files={"file": ("cv-2026.txt", changed, "text/plain")}).json()
    assert new["reused"] is False and len(calls) == 3
    assert sorted(p.name for p in (store.root / "resume").iterdir()) == ["cv-2026.txt"]


def test_resume_endpoints_before_upload(client):
    assert client.get("/api/resume").status_code == 404
    assert client.get("/api/resume/file").status_code == 404
    assert client.post("/api/resume/reanalyze").status_code == 404


# ------------------------------------------------------------------ company insights
def test_company_endpoints(client, monkeypatch):
    from app import main as m
    from app.models import CompanyInsight
    from app.services.gemini import GeminiError

    links = client.get("/api/company/links", params={"company": "Zoho", "role": "Data Engineer"}).json()
    assert [l["label"] for l in links][:2] == ["AmbitionBox reviews", "Glassdoor reviews"]
    assert client.get("/api/company/links", params={"company": " "}).status_code == 400

    body = {"company": "Zoho", "role": "Data Engineer"}
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    no_key = client.post("/api/company/insights", json=body).json()
    assert no_key["insight"] is None and "GEMINI_API_KEY" in no_key["ai_error"] and len(no_key["links"]) == 5

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    calls = []

    async def fake(company, role, location, *, api_key, model):
        calls.append(company)
        if company == "Broken":
            raise GeminiError("Gemini is overloaded right now")
        return CompanyInsight(company=company, role=role, grounded=False, overview=f"call {len(calls)}",
                              generated_at="2099-01-01T00:00:00+00:00")

    monkeypatch.setattr(m, "generate_insight", fake)
    first = client.post("/api/company/insights", json=body).json()
    assert first["insight"]["overview"] == "call 1" and first["ai_error"] == ""
    assert client.post("/api/company/insights", json=body).json()["insight"]["overview"] == "call 1"  # cached
    assert client.post("/api/company/insights", json={**body, "regenerate": True}).json()["insight"]["overview"] == "call 2"

    broken = client.post("/api/company/insights", json={"company": "Broken"})
    assert broken.status_code == 200  # AI failure never breaks the dialog
    assert broken.json()["insight"] is None and "overloaded" in broken.json()["ai_error"]
    assert broken.json()["links"]


def test_combined_view_endpoints(client):
    import io

    from openpyxl import load_workbook

    client.post("/api/resume", files={"file": ("cv.txt", SAMPLE.encode(), "text/plain")})
    client.post("/api/search", json={})
    client.post("/api/search", json={})  # same job found twice

    c = client.get("/api/searches/combined?days=3").json()
    assert len(c["searches"]) == 2 and c["total_listings"] == 2 and c["duplicates_removed"] == 1
    job = c["response"]["jobs"][0]
    assert job["company"] == "Zoho" and job["appearances"] == 2

    assert client.get("/api/searches/combined?days=0").status_code == 422
    assert client.get("/api/searches/combined?days=91").status_code == 422

    x = client.get("/api/searches/combined/export?days=3&format=xlsx")
    assert x.status_code == 200 and "last-3-days" in x.headers["content-disposition"]
    ws = load_workbook(io.BytesIO(x.content))["Matches"]
    headers = [cell.value for cell in ws[1]]
    assert ws.cell(2, headers.index("Times seen") + 1).value == 2
    assert "Zoho" in client.get("/api/searches/combined/export?days=1&format=csv").content.decode("utf-8-sig")
