from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest

from app.models import CompanyInsight
from app.services import company as company_mod
from app.services.company import InsightCache, _extract_json, _InsightOut, generate_insight, quick_links
from app.services.gemini import GeminiError, GroundedText, WebSource
from app.services.storage import Storage


def _query(url: str) -> str:
    return parse_qs(urlparse(url).query)["q"][0]


def test_quick_links_are_scoped_google_searches():
    links = {l.label: l.url for l in quick_links("Zoho Corporation", "Data Engineer", "Chennai")}
    assert _query(links["AmbitionBox reviews"]) == '"Zoho Corporation" reviews site:ambitionbox.com'
    assert _query(links["Glassdoor reviews"]) == '"Zoho Corporation" reviews site:glassdoor.co.in'
    assert _query(links["Data Engineer salary"]) == '"Zoho Corporation" Data Engineer salary Chennai India'
    assert all(u.startswith("https://www.google.com/search?q=") for u in links.values())
    # Odd characters can't break out of the query string.
    evil = quick_links('A&B "Co" #1', "")[0].url
    assert _query(evil) == '"A&B "Co" #1" reviews site:ambitionbox.com'


@pytest.mark.parametrize(
    "text",
    ['{"known": true}', '```json\n{"known": true}\n```', 'Here you go:\n{"known": true}\nHope this helps'],
)
def test_extract_json(text):
    assert _extract_json(text) == {"known": True}


def test_extract_json_rejects_prose():
    with pytest.raises(ValueError):
        _extract_json("I could not find this company.")


GROUNDED_JSON = """```json
{"known": true, "overview": "Zoho builds SaaS tools.", "rating": 4.04, "rating_source": "AmbitionBox",
 "review_count": "9,000 reviews", "pros": ["- Learning", "Learning", "Work-life balance"], "cons": ["Pay"],
 "salary_range": "INR 7-14 LPA", "salary_note": "AmbitionBox", "questions_to_ask": ["Q1", "Q2", "Q3", "Q4"],
 "caveats": ""}
```"""


async def test_grounded_insight_with_sources(monkeypatch):
    async def grounded(**_):
        return GroundedText(text=GROUNDED_JSON, sources=[
            WebSource("ambitionbox.com", "https://vertexaisearch.example/1"),
            WebSource("ambitionbox.com", "https://vertexaisearch.example/1"),  # duplicate URL
            WebSource("glassdoor.co.in", "https://vertexaisearch.example/2"),
        ])

    async def knowledge(**_):
        raise AssertionError("fallback must not run when live search works")

    monkeypatch.setattr(company_mod, "generate_grounded", grounded)
    monkeypatch.setattr(company_mod, "generate_json", knowledge)
    i = await generate_insight("Zoho", "Data Engineer", "Chennai", api_key="k", model="m")
    assert i.grounded and i.rating == 4.0 and i.rating_source == "AmbitionBox"
    assert i.pros == ["Learning", "Work-life balance"] and len(i.questions_to_ask) == 3
    assert [s.label for s in i.sources] == ["ambitionbox.com", "glassdoor.co.in"]


@pytest.mark.parametrize(
    "failure",
    [GeminiError("Gemini quota / rate limit reached"), "not json at all"],
)
async def test_falls_back_to_model_knowledge(monkeypatch, failure):
    prompts = []

    async def grounded(**kw):
        prompts.append(("grounded", kw["prompt"]))
        if isinstance(failure, Exception):
            raise failure
        return GroundedText(text=failure, sources=[])

    async def knowledge(*, prompt, schema, **_):
        prompts.append(("knowledge", prompt))
        return _InsightOut(known=False, overview="Small IT firm.", rating=9.5, rating_source="?",
                           salary_range="INR 3-6 LPA")

    monkeypatch.setattr(company_mod, "generate_grounded", grounded)
    monkeypatch.setattr(company_mod, "generate_json", knowledge)
    i = await generate_insight("Blob Infotech", "", "Chennai", api_key="k", model="m")
    assert not i.grounded and not i.known and i.sources == []
    assert i.rating is None and i.rating_source == ""  # impossible rating discarded
    assert i.role == ""  # cache key stays on the requested role
    assert [p[0] for p in prompts] == ["grounded", "knowledge"]
    assert "NO live data" in prompts[1][1] and "software professional" in prompts[1][1]


def _insight(grounded: bool, days_old: float, role: str = "Data Engineer") -> CompanyInsight:
    at = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    return CompanyInsight(company="Zoho Corporation", role=role, grounded=grounded, generated_at=at)


@pytest.mark.parametrize(
    ("grounded", "days_old", "hit"),
    [(True, 29, True), (True, 31, False), (False, 6, True), (False, 8, False)],
)
def test_cache_ttl(tmp_path, grounded, days_old, hit):
    cache = InsightCache(Storage(tmp_path))
    cache.put(_insight(grounded, days_old))
    assert (cache.get("Zoho Corporation", "Data Engineer") is not None) == hit


def test_cache_key_normalises_company_and_role(tmp_path):
    cache = InsightCache(Storage(tmp_path))
    cache.put(_insight(True, 0, role="Sr. Data Engineer"))
    assert cache.get("ZOHO Corp", "Senior Data Engineer") is not None
    assert cache.get("Zoho", "Data Scientist") is None
