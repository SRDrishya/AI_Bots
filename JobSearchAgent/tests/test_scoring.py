from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.models import JobListing, ResumeProfile
from app.services import scoring
from app.services.gemini import GeminiError
from app.services.scoring import (
    _GeminiBatch,
    _GeminiJobScore,
    job_level,
    level_fit,
    score_jobs,
    score_local,
    tfidf_similarities,
    title_fit,
    years_required,
)

PROFILE = ResumeProfile(
    candidate_name="Priya",
    summary="Data engineer building batch and streaming pipelines on AWS.",
    target_roles=["Data Engineer", "Analytics Engineer"],
    skills=["Python", "SQL", "Spark", "Airflow", "Kafka", "AWS", "Snowflake"],
    years_of_experience=5,
    seniority="Mid",
    preferred_locations=["Chennai", "Remote"],
    search_queries=["Data Engineer", "Python Data Engineer"],
)

_n = 0


def job(title, description="", location_match="local", posted_days_ago=None, **kw) -> JobListing:
    global _n
    _n += 1
    posted = ""
    if posted_days_ago is not None:
        posted = (datetime.now(timezone.utc) - timedelta(days=posted_days_ago)).isoformat()
    return JobListing(id=f"t:{_n}", title=title, company=kw.pop("company", "Acme"), url=f"https://x/{_n}",
                      source="jsearch", description=description, location_match=location_match,
                      posted_at=posted, **kw)


GOOD = job("Data Engineer", "Build pipelines with Python, Spark, Airflow and Kafka on AWS. SQL and Snowflake. 3+ years experience.")
OK = job("Software Engineer", "Java microservices with Spring Boot, some Python and SQL. Docker, Kubernetes.")
BAD = job("Account Executive", "Sell SaaS to enterprise customers. Salesforce CRM. Quota carrying.")
HIGH = job("Principal Data Engineer", "Python, Spark, Airflow, AWS. 12+ years of experience required.")


# ------------------------------------------------------------------ components
def test_title_fit():
    roles = ["Data Engineer", "Software Engineer"]
    assert title_fit("Senior Data Engineer", roles) == (1.0, "Data Engineer")
    assert title_fit("Data Platform Developer", roles)[0] == 1.0  # developer == engineer
    assert title_fit("Data Scientist", roles)[0] == 0.5
    assert title_fit("Backend Engineer", ["Software Engineer"])[0] == 0.75
    assert title_fit("Account Executive", roles)[0] == 0.0


@pytest.mark.parametrize(
    ("title", "level"),
    [("Software Engineering Intern", 0), ("Junior Data Analyst", 1), ("Data Engineer", None),
     ("Sr. Data Engineer", 3), ("Senior Data Engineer", 3), ("Lead Data Engineer", 4),
     ("Principal Engineer", 4), ("Engineering Manager", 4)],
)
def test_job_level(title, level):
    assert job_level(title) == level


@pytest.mark.parametrize(
    ("text", "years"),
    [("Requires 5+ years of experience", 5), ("3-5 yrs exp in Python", 3), ("Experience: 4 to 8 years", 4),
     ("Founded 150 years ago", None), ("Version 2.5 years", None), ("No requirement mentioned", None)],
)
def test_years_required(text, years):
    assert years_required(text) == years


def test_level_fit_penalises_over_reach():
    factor, notes = level_fit(HIGH, PROFILE)
    assert factor < 0.5
    assert "role is well above your level" in notes and "asks 12+ yrs (you: 5)" in notes
    assert level_fit(GOOD, PROFILE) == (1.0, ["asks 3+ yrs ✓"])


def test_tfidf_ranks_relevant_doc_first():
    sims = tfidf_similarities("python spark airflow data pipelines", [
        "We build data pipelines with Spark and Airflow in Python",
        "Sell enterprise software to customers",
    ])
    assert sims[0] > 0.2 and sims[1] == 0.0


# ------------------------------------------------------------------ local scoring
def test_score_local_orders_and_explains():
    scored = score_local([BAD, OK, HIGH, GOOD], PROFILE)
    assert [j.title for j in scored][0] == "Data Engineer"
    assert scored[-1].title == "Account Executive"
    by_title = {j.title: j for j in scored}
    good = by_title["Data Engineer"]
    assert good.match_score >= 80
    assert {"Python", "Spark", "Airflow", "Kafka", "AWS", "SQL", "Snowflake"} <= set(good.matched_skills)
    assert good.match_reasoning.startswith("Title matches 'Data Engineer'")
    assert by_title["Principal Data Engineer"].match_score < good.match_score - 30
    assert "Spring Boot" in by_title["Software Engineer"].missing_skills
    assert all(j.scoring_engine == "local" for j in scored)
    assert all(0 <= j.match_score <= 100 for j in scored)


def test_bonuses_for_local_and_recent():
    desc = "Python, Java, Scala, Rust and Go."  # 1/5 skills -> well below the 100 cap
    a = job("Data Engineer", desc, location_match="remote")
    b = job("Data Engineer", desc, location_match="local", posted_days_ago=2)
    scores = {j.id: j.match_score for j in score_local([a, b], PROFILE)}
    assert scores[b.id] - scores[a.id] == scoring.LOCAL_BONUS + scoring.RECENT_BONUS


def test_score_local_empty():
    assert score_local([], PROFILE) == []


# ------------------------------------------------------------------ gemini re-ranking
async def test_no_key_is_local():
    result = await score_jobs([GOOD, BAD], PROFILE)
    assert result.engine == "local"


async def test_gemini_rescores_top_n_only(monkeypatch):
    calls = []

    async def fake_generate_json(*, prompt, schema, **_):
        calls.append(prompt)
        ids = [line.split("id: ")[1] for line in prompt.splitlines() if line.startswith("--- id: ")]
        return _GeminiBatch(results=[
            _GeminiJobScore(id=i, score=40, matched_skills=["pyspark"], missing_skills=["dbt"], reasoning=f"AI {i}")
            for i in ids
        ] + [_GeminiJobScore(id="j999", score=100)])  # unknown id is ignored

    monkeypatch.setattr(scoring, "generate_json", fake_generate_json)
    jobs = [job(f"Data Engineer {i}", "Python Spark Airflow") for i in range(23)]
    result = await score_jobs(jobs, PROFILE, gemini_api_key="k", gemini_model="m", top_n=21)

    assert result.engine == "gemini"
    assert len(calls) == 3  # 21 jobs -> batches of 10, 10, 1
    assert "Target roles: Data Engineer, Analytics Engineer" in calls[0]
    ai = [j for j in result.jobs if j.scoring_engine == "gemini"]
    assert len(ai) == 21
    sample = ai[0]
    assert sample.matched_skills == ["Spark"] and sample.missing_skills == ["dbt"]
    assert sample.match_reasoning.startswith("AI j")
    local_score = next(j for j in score_local(jobs, PROFILE) if j.id == sample.id).match_score
    assert sample.match_score == round(0.75 * 40 + 0.25 * local_score)
    # The 2 jobs Gemini didn't see keep their (now higher) local score and rank first.
    assert [j.scoring_engine for j in result.jobs[:2]] == ["local", "local"]


async def test_gemini_total_failure_falls_back(monkeypatch):
    async def boom(**_):
        raise GeminiError("Gemini quota / rate limit reached")

    monkeypatch.setattr(scoring, "generate_json", boom)
    result = await score_jobs([GOOD, BAD], PROFILE, gemini_api_key="k", gemini_model="m")
    assert result.engine == "local (gemini failed: Gemini quota / rate limit reached)"
    assert [j.scoring_engine for j in result.jobs] == ["local", "local"]


async def test_gemini_partial_failure(monkeypatch):
    calls = 0

    async def flaky(*, prompt, **_):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise GeminiError("Gemini timed out")
        ids = [line.split("id: ")[1] for line in prompt.splitlines() if line.startswith("--- id: ")]
        return _GeminiBatch(results=[_GeminiJobScore(id=i, score=90, reasoning="ok") for i in ids])

    monkeypatch.setattr(scoring, "generate_json", flaky)
    jobs = [job(f"Data Engineer {i}", "Python") for i in range(15)]
    result = await score_jobs(jobs, PROFILE, gemini_api_key="k", gemini_model="m", top_n=15)
    assert result.engine == "gemini (partial: Gemini timed out)"
    assert sum(j.scoring_engine == "gemini" for j in result.jobs) == 10
