from __future__ import annotations

import asyncio

import httpx
import pytest

import app.sources as sources_mod
from app.models import ApiKeysInput, ATSBoardConfig, ResumeProfile, SearchRequest
from app.sources import fetch_all
from app.sources.ats import fetch_ats_boards
from app.sources.base import (
    SearchContext,
    SourceSkipped,
    TitleMatcher,
    clear_cache,
    format_salary,
    html_to_text,
    iso_date,
)
from app.sources.free import fetch_himalayas, fetch_jobicy, fetch_remotive
from app.sources.paid import fetch_adzuna, fetch_jsearch, fetch_serpapi

QUERIES = ["Data Engineer", "Python Data Engineer", "Software Engineer"]


@pytest.fixture(autouse=True)
def _isolate():
    clear_cache()  # (real .env keys are blocked in conftest.py)
    yield
    clear_cache()


def make_ctx(**overrides) -> SearchContext:
    base = dict(
        queries=QUERIES,
        location="Chennai",
        remote_only=False,
        keys=ApiKeysInput(),
        ats_boards=[],
        per_source_limit=100,
        matcher=TitleMatcher(QUERIES, ["Data Engineer"], ["Python", "Spark"]),
    )
    base.update(overrides)
    return SearchContext(**base)


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------ helpers
def test_html_to_text_handles_greenhouse_double_escaping():
    assert html_to_text("&lt;p&gt;Build &amp;amp; ship&lt;/p&gt;&lt;li&gt;Python&lt;/li&gt;") == "Build & ship\nPython"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1795364410, "2026-11-22T"),
        (1792767640768, "2026-10-23T15:00:40"),  # milliseconds
        ("2026-11-19T12:12:24-05:00", "2026-11-19T17:12:24+00:00"),
        ("2026-11-19T14:58:42", "2026-11-19T14:58:42+00:00"),
        ("sometime last spring", "sometime last spring"),  # unparseable -> kept
        (None, ""),
    ],
)
def test_iso_date(value, expected):
    assert iso_date(value).startswith(expected)


def test_iso_date_relative():
    from datetime import datetime, timedelta, timezone

    got = datetime.fromisoformat(iso_date("12 days ago"))
    expected = datetime.now(timezone.utc) - timedelta(days=12)
    assert abs((got - expected).total_seconds()) < 5
    assert iso_date("30+ days ago") < iso_date("3 hours ago") <= iso_date("Today")


async def test_serpapi_mapping():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        return httpx.Response(200, json={"jobs_results": [{
            "job_id": "abc", "title": "Data and Analytics Engineer", "company_name": "CITI",
            "location": "Chennai, Tamil Nadu", "via": "Citi Careers", "description": "SQL, Python",
            "detected_extensions": {"posted_at": "12 days ago", "schedule_type": "Full–time"},
            "apply_options": [{"title": "Citi Careers", "link": "https://jobs.citi.com/1"}],
        }, {
            "job_id": "def", "title": "Senior Data Engineer", "company_name": "Kyndryl", "location": "Anywhere",
            "detected_extensions": {"work_from_home": True}, "share_link": "https://google/x",
        }]})

    async with mock_client(handler) as client:
        jobs = await fetch_serpapi(make_ctx(keys=ApiKeysInput(serpapi_key="k")), client)

    assert [(p["q"], p["location"]) for p in seen] == [
        ("Data Engineer", "Chennai, Tamil Nadu, India"),
        ("Python Data Engineer", "Chennai, Tamil Nadu, India"),
        ("Data Engineer remote", "India"),
    ]
    citi, kyndryl = jobs
    assert citi.url == "https://jobs.citi.com/1" and citi.employment_type == "Full-time"
    assert citi.posted_at[:4] == "2026" and not citi.remote
    assert kyndryl.remote and kyndryl.location == "Remote (Anywhere)" and kyndryl.url == "https://google/x"


def test_format_salary():
    assert format_salary(1_200_000, 1_800_000, "inr", "year") == "INR 12.0L - 18.0L / year"
    assert format_salary(96300, 120400, "GBP") == "GBP 96K - 120K"
    assert format_salary(None, None) == ""


def test_title_matcher():
    m = TitleMatcher(["Data Engineer"], ["Software Engineer"], ["Python", "Apache Flink"])
    assert m.matches("Senior Data Engineer")
    assert m.matches("Data Platform Engineer, Payments")
    assert m.matches("Backend Developer")  # broad software role
    assert m.matches("Python Specialist")  # top skill in title
    assert not m.matches("Account Executive")
    assert not m.matches("Senior Customer Success - EMEA")
    assert not m.matches("Senior Data Scientist (Applied ML)")
    assert not m.matches("Group Product Manager, Data Feeds")


def test_title_matcher_synonyms():
    m = TitleMatcher(["Backend Developer"], [], [])
    assert m.matches("Back End Engineer")
    assert m.matches("Sr. Back-end Developer (Go)")
    assert not m.matches("Frontend Developer")


# ------------------------------------------------------------------ free sources
async def test_remotive_filters_titles_and_caches():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"jobs": [
            {"id": 1, "url": "https://remotive.com/1", "title": "Senior Data Engineer", "company_name": "Acme",
             "category": "Data", "tags": ["spark"], "job_type": "full_time",
             "publication_date": "2026-11-19T14:58:42", "candidate_required_location": "Worldwide",
             "salary": "$100k", "description": "<p>Spark &amp; Airflow</p>"},
            {"id": 2, "url": "https://remotive.com/2", "title": "Content Reviewer", "company_name": "X"},
        ]})

    async with mock_client(handler) as client:
        jobs = await fetch_remotive(make_ctx(), client)
        await fetch_remotive(make_ctx(), client)

    assert calls == 1  # second call served from cache
    assert [j.title for j in jobs] == ["Senior Data Engineer"]
    job = jobs[0]
    assert job.id == "remotive:1" and job.location == "Remote (Worldwide)" and job.remote
    assert job.employment_type == "Full Time" and job.description == "Spark & Airflow"


async def test_himalayas_sends_india_filter_and_dedupes():
    seen_params = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.append(dict(request.url.params))
        return httpx.Response(200, json={"jobs": [{
            "title": "Data Engineer", "companyName": "Drivetrain", "employmentType": "Full Time",
            "minSalary": 2000, "maxSalary": 3000, "currency": "USD", "salaryPeriod": "monthly",
            "seniority": ["Senior"], "locationRestrictions": ["India", "United States"],
            "categories": ["Data-Engineer"], "description": "<p>Hi</p>", "pubDate": 1793013430,
            "applicationLink": "https://himalayas.app/companies/drivetrain/jobs/data-engineer",
            "guid": "https://himalayas.app/companies/drivetrain/jobs/data-engineer",
        }]})

    async with mock_client(handler) as client:
        jobs = await fetch_himalayas(make_ctx(), client)

    assert all(p["country"] == "India" for p in seen_params)
    assert [p["q"] for p in seen_params] == QUERIES
    assert len(jobs) == 1  # same job from 3 queries
    assert jobs[0].location == "Remote (India, United States)"
    assert jobs[0].salary == "USD 2K - 3K / monthly"


async def test_jobicy_filters_loose_tag_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jobs": [
            {"id": 1, "url": "u1", "jobTitle": "Data Engineer", "companyName": "A", "jobGeo": "APAC",
             "jobType": ["Full-Time"], "jobLevel": "Senior", "jobIndustry": ["Data"], "pubDate": "2026-11-22T11:58:12+00:00"},
            {"id": 2, "url": "u2", "jobTitle": "Senior Customer Success - EMEA", "companyName": "B", "jobGeo": "Europe"},
        ]})

    async with mock_client(handler) as client:
        jobs = await fetch_jobicy(make_ctx(), client)
    assert [(j.title, j.location) for j in jobs] == [("Data Engineer", "Remote (APAC)")]


# ------------------------------------------------------------------ ATS boards
async def test_ats_boards_partial_failure_and_priority():
    def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if host == "boards-api.greenhouse.io":
            return httpx.Response(200, json={"jobs": [
                {"id": 1, "title": "Data Engineer", "company_name": "Stripe", "location": {"name": "Dublin"},
                 "absolute_url": "https://stripe.com/1", "updated_at": "2026-11-19T12:12:24-05:00",
                 "content": "&lt;p&gt;Hi&lt;/p&gt;", "departments": [{"name": "Data"}]},
                {"id": 2, "title": "Account Executive", "location": {"name": "Chennai"}, "absolute_url": "x"},
                {"id": 3, "title": "Senior Data Engineer", "location": {"name": "Chennai, India"}, "absolute_url": "y"},
            ]})
        if host == "api.lever.co":
            return httpx.Response(200, json=[{
                "id": "abc", "text": "Data Engineer", "hostedUrl": "https://jobs.lever.co/x/abc",
                "createdAt": 1792767640768, "workplaceType": "remote",
                "categories": {"location": "India", "allLocations": ["India"], "commitment": "Permanent"},
                "descriptionPlain": "Build pipelines",
            }])
        if "brokenco" in path:
            return httpx.Response(404)
        return httpx.Response(200, json={"jobs": [{
            "id": "z", "title": "Data Engineer", "location": "San Francisco", "workplaceType": "Hybrid",
            "isRemote": True, "isListed": True, "jobUrl": "https://jobs.ashbyhq.com/z",
            "compensation": {"scrapeableCompensationSalarySummary": "$200K - $250K"},
        }]})

    boards = [
        ATSBoardConfig(provider="greenhouse", slug="stripe", label="Stripe"),
        ATSBoardConfig(provider="lever", slug="x", label="X"),
        ATSBoardConfig(provider="ashby", slug="openai", label="OpenAI"),
        ATSBoardConfig(provider="ashby", slug="brokenco", label="BrokenCo"),
        ATSBoardConfig(provider="greenhouse", slug="off", enabled=False),
    ]
    async with mock_client(handler) as client:
        jobs, message = await fetch_ats_boards(make_ctx(ats_boards=boards), client)

    assert message == "3/4 boards ok; failed: BrokenCo (HTTP 404)"
    titles_locs = [(j.source, j.location) for j in jobs]
    assert titles_locs[0] == ("greenhouse", "Chennai, India")  # city first
    assert titles_locs[1] == ("lever", "India")
    assert "Account Executive" not in [j.title for j in jobs]
    ashby = next(j for j in jobs if j.source == "ashby")
    assert ashby.remote is False  # Hybrid workplace beats isRemote flag
    assert ashby.salary == "$200K - $250K" and ashby.company == "OpenAI"


async def test_ats_all_boards_fail():
    async with mock_client(lambda r: httpx.Response(500)) as client:
        with pytest.raises(RuntimeError, match="All 1 boards failed"):
            await fetch_ats_boards(make_ctx(ats_boards=[ATSBoardConfig(provider="lever", slug="a")]), client)


# ------------------------------------------------------------------ paid sources
@pytest.mark.parametrize("fn", [fetch_adzuna, fetch_jsearch, fetch_serpapi])
async def test_paid_sources_skip_without_keys(fn):
    async with mock_client(lambda r: httpx.Response(500)) as client:
        with pytest.raises(SourceSkipped):
            await fn(make_ctx(), client)


async def test_jsearch_query_plan_and_mapping():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        assert request.headers["X-RapidAPI-Key"] == "k"
        assert request.url.path == "/search-v2"
        return httpx.Response(200, json={"status": "OK", "data": {"cursor": "c", "jobs": [{
            "job_id": f"id-{len(seen)}", "job_title": "Data Engineer", "employer_name": "Zoho",
            "job_city": "Chennai", "job_state": "Tamil Nadu", "job_country": "IN", "job_is_remote": False,
            "job_apply_link": "https://apply", "job_posted_at_datetime_utc": "2026-11-20T00:00:00.000Z",
            "job_employment_types": ["FULLTIME"], "job_publisher": "Naukri", "job_description": "Build stuff",
        }]}})

    ctx = make_ctx(keys=ApiKeysInput(rapidapi_key="k"))
    async with mock_client(handler) as client:
        jobs = await fetch_jsearch(ctx, client)

    assert [p["query"] for p in seen] == [
        "Data Engineer in Chennai, India",
        "Python Data Engineer in Chennai, India",
        "Data Engineer in India",
    ]
    assert seen[2]["work_from_home"] == "true" and "work_from_home" not in seen[0]
    assert len(jobs) == 3
    assert jobs[0].location == "Chennai, Tamil Nadu, IN" and jobs[0].tags == ["via Naukri"]
    assert jobs[0].employment_type == "Full Time"
    assert all(p["country"] == "in" for p in seen)


async def test_adzuna_uses_india_endpoint_and_city():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json={"results": [{
            "id": 42, "title": "<strong>Data</strong> Engineer", "description": "Python, Spark",
            "created": "2026-11-20T10:00:00Z", "redirect_url": "https://adzuna/42",
            "company": {"display_name": "Freshworks"}, "location": {"display_name": "Chennai, Tamil Nadu"},
            "salary_min": 1200000, "salary_max": 1800000, "salary_is_predicted": "1",
            "contract_time": "full_time", "category": {"label": "IT Jobs"},
        }]})

    ctx = make_ctx(keys=ApiKeysInput(adzuna_app_id="i", adzuna_app_key="k"))
    async with mock_client(handler) as client:
        jobs = await fetch_adzuna(ctx, client)

    assert all(u.path == "/v1/api/jobs/in/search/1" and u.params["where"] == "Chennai" for u in seen)
    assert len(seen) == 2 and len(jobs) == 1  # same id deduped
    assert jobs[0].title == "Data Engineer"
    assert jobs[0].salary == "INR 12.0L - 18.0L / year (estimated)"


async def test_serpapi_error_and_empty_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "Google hasn't returned any results for this query."})

    async with mock_client(handler) as client:
        assert await fetch_serpapi(make_ctx(keys=ApiKeysInput(serpapi_key="k")), client) == []

    async with mock_client(lambda r: httpx.Response(200, json={"error": "Invalid API key."})) as client:
        with pytest.raises(RuntimeError, match="Invalid API key"):
            await fetch_serpapi(make_ctx(keys=ApiKeysInput(serpapi_key="k")), client)


# ------------------------------------------------------------------ aggregator
async def test_fetch_all_reports_every_outcome(monkeypatch):
    async def ok(ctx, client):
        return []

    async def rate_limited(ctx, client):
        request = httpx.Request("GET", "https://api.example.com")
        raise httpx.HTTPStatusError("429", request=request, response=httpx.Response(429, request=request))

    async def slow(ctx, client):
        await asyncio.sleep(5)

    monkeypatch.setattr(sources_mod, "SOURCES", {"good": ok, "limited": rate_limited, "slow": slow,
                                                 "jsearch": sources_mod.SOURCES["jsearch"]})
    monkeypatch.setattr(sources_mod, "SOURCE_TIMEOUT", 0.05)

    request = SearchRequest(
        profile=ResumeProfile(search_queries=["Data Engineer"]),
        sources=["good", "limited", "slow", "jsearch", "nope", "good"],
    )
    async with mock_client(lambda r: httpx.Response(500)) as client:
        jobs, statuses = await fetch_all(request, client)

    got = {s.source: (s.status, s.message) for s in statuses}
    assert jobs == [] and len(statuses) == 5  # duplicate "good" collapsed
    assert got["good"] == ("ok", "")
    assert got["limited"] == ("error", "Rate limit / quota reached (HTTP 429)")
    assert got["slow"] == ("error", "Timed out")
    assert got["jsearch"][0] == "skipped"
    assert got["nope"] == ("error", "Unknown source")


async def test_fetch_all_requires_search_terms():
    with pytest.raises(ValueError, match="No search terms"):
        await fetch_all(SearchRequest(profile=ResumeProfile()))


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Senior Data Engineer Chennai", "Senior Data Engineer"),
        ("Python Data Engineer Remote", "Python Data Engineer"),
        ("Data Engineer jobs in Bengaluru, India", "Data Engineer"),
        ("Snowflake Data Engineer - OMR", "Snowflake Data Engineer"),
        ("Remote", "Remote"),  # never cleans to nothing
        ("C++ Developer", "C++ Developer"),
    ],
)
def test_clean_query(raw, clean):
    from app.sources.base import clean_query

    assert clean_query(raw) == clean


def test_build_queries_cleans_and_dedupes():
    from app.sources.base import build_queries

    profile = ResumeProfile(search_queries=["Data Engineer Chennai", "Data Engineer Remote", "AWS Data Engineer"])
    assert build_queries(profile, "") == ["Data Engineer", "AWS Data Engineer"]
    assert build_queries(profile, "ML Engineer in Chennai; NLP Engineer") == ["ML Engineer", "NLP Engineer"]


def test_request_keys_override_env(monkeypatch):
    monkeypatch.setenv("SERPAPI_KEY", "from-env")
    monkeypatch.setenv("RAPIDAPI_KEY", "env-rapid")
    ctx = sources_mod.build_context(SearchRequest(
        profile=ResumeProfile(search_queries=["x"]),
        api_keys=ApiKeysInput(serpapi_key="from-request"),
    ))
    assert ctx.keys.serpapi_key == "from-request"
    assert ctx.keys.rapidapi_key == "env-rapid"
    assert ctx.city == "Chennai"
    assert len(ctx.ats_boards) == 10  # falls back to config list


# ------------------------------------------------------------------ quota guard
class _MemDocs:
    def __init__(self):
        self.docs = {}

    def load_doc(self, name, default):
        return self.docs.get(name, default)

    def save_doc(self, name, data):
        self.docs[name] = data


def _jsearch_handler(remaining_start: int, seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["query"])
        left = remaining_start - len(seen)
        return httpx.Response(200, headers={"x-ratelimit-requests-remaining": str(left),
                                            "x-ratelimit-requests-reset": "86400"},
                              json={"data": {"jobs": []}})
    return handler


async def test_quota_guard_records_and_trims_jsearch():
    from app.sources.quota import QuotaTracker

    quota = QuotaTracker(_MemDocs())
    seen: list = []
    ctx = make_ctx(keys=ApiKeysInput(rapidapi_key="k"), quota=quota)
    async with mock_client(_jsearch_handler(4, seen)) as client:
        await fetch_jsearch(ctx, client)          # unknown quota -> full plan (3 calls)
        assert len(seen) == 3 and quota.remaining("jsearch") == 1
        await fetch_jsearch(ctx, client)          # only 1 left -> 1 call
        assert len(seen) == 4 and quota.remaining("jsearch") == 0
        with pytest.raises(SourceSkipped, match="quota used up; resets"):
            await fetch_jsearch(ctx, client)      # nothing left -> no call at all
    assert len(seen) == 4


def test_quota_unknown_after_reset_passes():
    from app.sources.quota import QuotaTracker

    quota = QuotaTracker(_MemDocs())
    quota.update("jsearch", remaining=0, reset_at="2000-01-01T00:00:00+00:00")
    assert quota.remaining("jsearch") is None and quota.allow("jsearch", 3) == 3
    quota.update("serpapi", remaining=10)
    quota.update("serpapi", used=3)
    assert quota.remaining("serpapi") == 7
    assert QuotaTracker(quota.store, reserve=5).allow("serpapi", 3) == 2


async def test_serpapi_syncs_from_account_endpoint():
    from app.sources.quota import QuotaTracker

    quota = QuotaTracker(_MemDocs())
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/account.json":
            return httpx.Response(200, json={"total_searches_left": 2, "plan_searches_left": 2})
        return httpx.Response(200, json={"jobs_results": []})

    async with mock_client(handler) as client:
        await fetch_serpapi(make_ctx(keys=ApiKeysInput(serpapi_key="k"), quota=quota), client)
    assert calls == ["/account.json", "/search.json", "/search.json"]  # trimmed from 3 to 2
    assert quota.remaining("serpapi") == 0
