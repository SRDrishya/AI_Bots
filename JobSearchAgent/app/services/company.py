"""Company insights: reviews, rating and pay for an employer — without spending job-portal credits.

1. Quick links (always, free): Google searches scoped to AmbitionBox, Glassdoor and LinkedIn,
   plus a salary search — they open the live pages, so they're never stale.
2. AI summary (on demand, cached):
     a. Gemini + Google Search  -> live data with source links ("grounded")
     b. if search grounding isn't available on the key/plan, Gemini's own knowledge,
        clearly labelled as an unverified estimate.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

from pydantic import BaseModel, Field, ValidationError

from app.models import CompanyInsight, CompanyLink
from app.services.filtering import normalize_company, normalize_title
from app.services.gemini import GeminiError, generate_grounded, generate_json
from app.services.storage import Storage, now_iso

log = logging.getLogger(__name__)

GROUNDED_TTL = timedelta(days=30)
ESTIMATE_TTL = timedelta(days=7)
MAX_CACHED = 200
MAX_SOURCES = 8


# --------------------------------------------------------------------------- quick links
def _google(query: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(query)}"


def quick_links(company: str, role: str = "", location: str = "Chennai") -> list[CompanyLink]:
    name = company.strip()
    role_part = f"{role.strip()} " if role.strip() else ""
    return [
        CompanyLink(label="AmbitionBox reviews", url=_google(f'"{name}" reviews site:ambitionbox.com')),
        CompanyLink(label="Glassdoor reviews", url=_google(f'"{name}" reviews site:glassdoor.co.in')),
        CompanyLink(label=f"{role_part}salary", url=_google(f'"{name}" {role_part}salary {location} India')),
        CompanyLink(label="LinkedIn page", url=_google(f'"{name}" site:linkedin.com/company')),
        CompanyLink(label="Recent news", url=_google(f'"{name}" India news')),
    ]


# --------------------------------------------------------------------------- AI summary
class _InsightOut(BaseModel):
    known: bool = True
    overview: str = ""
    rating: float | None = None
    rating_source: str = ""
    review_count: str = ""
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    salary_range: str = ""
    salary_note: str = ""
    questions_to_ask: list[str] = Field(default_factory=list)
    caveats: str = ""


_FIELDS = """Fields:
- known: false if you can't find / don't know this specific employer (then leave the rest mostly empty)
- overview: 1-2 sentences — what the company does, approximate size, presence in India
- rating: overall employee rating out of 5 (number) or null; rating_source: where it's from (e.g. "AmbitionBox")
- review_count: approximate number of reviews if known, e.g. "2,300 reviews", else ""
- pros / cons: 3-4 short recurring themes each from employee reviews
- salary_range: typical annual pay for the ROLE at this company in India, e.g. "INR 8-15 LPA", else ""
- salary_note: one line on what the range is based on and how it varies by experience
- questions_to_ask: 3 smart questions to ask in the interview, tailored to the company and role
- caveats: anything the candidate should double-check"""

_GROUNDED_PROMPT = """Research the employer "{company}" for a job seeker in {location}, India applying for "{role}".
Use Google Search. Prefer AmbitionBox and Glassdoor (India) for ratings, review themes and salaries,
and recent news for anything notable (layoffs, funding, growth). Never guess numbers — leave a field
empty if the sources don't state it. If several companies share this name, pick the one most likely
hiring a "{role}" in {location} and say so in caveats.

Reply with ONLY a JSON object (no markdown, no commentary).
{fields}"""

_KNOWLEDGE_PROMPT = """From your own general knowledge only (you have NO live data), describe the employer
"{company}" for a job seeker in {location}, India applying for "{role}".
Rules:
- If you don't reliably know this specific company, set known=false and leave rating, review_count
  and salary_range empty. Do not guess.
- Only give a rating / review count if you are confident it is approximately right; otherwise null / "".
- salary_range may be a typical market range for the role and company type in India if the company
  itself is unknown to you — say so in salary_note.
- caveats must mention that this is not based on live data.
{fields}"""


def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(cleaned[start:end + 1])


def _clean_list(items: list[str], limit: int) -> list[str]:
    return list(dict.fromkeys(i.strip(" -•*") for i in items if i and i.strip(" -•*")))[:limit]


def _build(company: str, role: str, out: _InsightOut, *, grounded: bool,
           sources: list[CompanyLink]) -> CompanyInsight:
    rating = out.rating if out.rating is not None and 1.0 <= out.rating <= 5.0 else None
    return CompanyInsight(
        company=company, role=role, grounded=grounded, known=out.known,
        overview=out.overview.strip(),
        rating=round(rating, 1) if rating is not None else None,
        rating_source=out.rating_source.strip() if rating is not None else "",
        review_count=out.review_count.strip(),
        pros=_clean_list(out.pros, 4), cons=_clean_list(out.cons, 4),
        salary_range=out.salary_range.strip(), salary_note=out.salary_note.strip(),
        questions_to_ask=_clean_list(out.questions_to_ask, 3),
        caveats=out.caveats.strip(),
        sources=sources[:MAX_SOURCES],
        generated_at=now_iso(),
    )


async def generate_insight(company: str, role: str, location: str, *, api_key: str, model: str) -> CompanyInsight:
    asked_role = role or "software professional"  # prompt wording; cache stays keyed on `role`
    try:
        grounded = await generate_grounded(
            api_key=api_key, model=model,
            prompt=_GROUNDED_PROMPT.format(company=company, role=asked_role, location=location, fields=_FIELDS))
        out = _InsightOut.model_validate(_extract_json(grounded.text))
        sources = list({s.url: CompanyLink(label=s.title, url=s.url) for s in grounded.sources}.values())
        return _build(company, role, out, grounded=True, sources=sources)
    except (GeminiError, ValueError, ValidationError) as exc:
        log.info("Live company research unavailable (%s); using model knowledge", exc)

    out = await generate_json(
        api_key=api_key, model=model, schema=_InsightOut, temperature=0.2,
        prompt=_KNOWLEDGE_PROMPT.format(company=company, role=asked_role, location=location, fields=_FIELDS))
    return _build(company, role, out, grounded=False, sources=[])


# --------------------------------------------------------------------------- cache
class InsightCache:
    DOC = "company_insights.json"

    def __init__(self, storage: Storage):
        self.storage = storage

    @staticmethod
    def key(company: str, role: str) -> str:
        return f"{normalize_company(company)}|{normalize_title(role or '')}"

    def get(self, company: str, role: str) -> CompanyInsight | None:
        raw = self.storage.load_doc(self.DOC, {}).get(self.key(company, role))
        if not raw:
            return None
        try:
            insight = CompanyInsight.model_validate(raw)
            age = datetime.now(timezone.utc) - datetime.fromisoformat(insight.generated_at)
        except (ValidationError, ValueError):
            return None
        return insight if age <= (GROUNDED_TTL if insight.grounded else ESTIMATE_TTL) else None

    def put(self, insight: CompanyInsight) -> None:
        docs = self.storage.load_doc(self.DOC, {})
        docs[self.key(insight.company, insight.role)] = insight.model_dump()
        if len(docs) > MAX_CACHED:
            docs = dict(sorted(docs.items(), key=lambda kv: kv[1].get("generated_at", ""))[-MAX_CACHED:])
        self.storage.save_doc(self.DOC, docs)
