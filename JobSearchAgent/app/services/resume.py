"""Resume ingestion: file -> text -> ResumeProfile.

Pipeline:
  1. extract_text()     PDF / DOCX / TXT bytes -> plain text
  2. analyze_local()    regex + skill-taxonomy heuristics (always runs, no network)
  3. analyze_gemini()   structured JSON extraction via Gemini (only if a key is set)
  4. merge              Gemini wins where it has data; local fills gaps.
                        Any Gemini failure falls back to the local profile.
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from datetime import date

from pydantic import BaseModel, Field

from app.config import DEFAULT_GEMINI_MODEL, DEFAULT_PREFERRED_LOCATIONS, MAX_RESUME_BYTES
from app.models import ResumeProfile
from app.services.gemini import generate_json
from app.services.skills import extract_roles, extract_skills, normalize_skill

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")
PREVIEW_CHARS = 1500
GEMINI_MAX_INPUT_CHARS = 30_000


class ResumeParseError(ValueError):
    """Raised when a resume can't be read; message is safe to show the user."""


# --------------------------------------------------------------------------- #
# 1. Text extraction
# --------------------------------------------------------------------------- #
def extract_text(filename: str, data: bytes) -> str:
    if not data:
        raise ResumeParseError("The uploaded file is empty.")
    if len(data) > MAX_RESUME_BYTES:
        raise ResumeParseError(f"File is larger than {MAX_RESUME_BYTES // (1024 * 1024)} MB.")

    name = (filename or "").lower()
    if name.endswith(".pdf"):
        text = _pdf_text(data)
    elif name.endswith(".docx"):
        text = _docx_text(data)
    elif name.endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="replace")
    elif name.endswith(".doc"):
        raise ResumeParseError("Old .doc files aren't supported. Save it as .docx or PDF and upload again.")
    else:
        raise ResumeParseError(f"Unsupported file type. Upload one of: {', '.join(SUPPORTED_EXTENSIONS)}")

    text = _clean_text(text)
    if len(text) < 50:
        raise ResumeParseError(
            "Couldn't find readable text in this file. If it's a scanned PDF, "
            "export it as a text-based PDF or DOCX."
        )
    return text


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # pypdf raises several types here
                raise ResumeParseError("This PDF is password-protected.") from exc
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except PdfReadError as exc:
        raise ResumeParseError("This PDF appears to be corrupted or unreadable.") from exc


def _docx_text(data: bytes) -> str:
    import docx
    from docx.opc.exceptions import PackageNotFoundError

    try:
        document = docx.Document(io.BytesIO(data))
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise ResumeParseError("This DOCX file appears to be corrupted or unreadable.") from exc

    parts = [p.text for p in document.paragraphs]
    # Many resume templates put skills / experience inside tables.
    for table in document.tables:
        for row in table.rows:
            seen: set[int] = set()
            for cell in row.cells:
                if id(cell._tc) in seen:  # merged cells repeat
                    continue
                seen.add(id(cell._tc))
                parts.append(cell.text)
    return "\n".join(parts)


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# 2. Local heuristic analysis
# --------------------------------------------------------------------------- #
_SECTION_HEADINGS = {
    "summary": ["summary", "professional summary", "profile", "objective", "about me", "career objective"],
    "experience": ["experience", "work experience", "professional experience", "employment history",
                   "work history", "employment"],
    "education": ["education", "academic qualifications", "qualifications", "academics"],
    "skills": ["skills", "technical skills", "core competencies", "key skills", "technologies"],
    "projects": ["projects", "key projects", "academic projects"],
    "certifications": ["certifications", "certificates", "achievements", "awards"],
}

_CITIES = [
    "Chennai", "Bangalore", "Bengaluru", "Hyderabad", "Pune", "Mumbai", "Delhi", "Noida",
    "Gurgaon", "Gurugram", "Kolkata", "Coimbatore", "Madurai", "Trichy", "Kochi", "Ahmedabad",
]
_CITY_CANONICAL = {"Bengaluru": "Bangalore", "Gurugram": "Gurgaon", "Madras": "Chennai"}


def split_sections(text: str) -> dict[str, str]:
    """Split resume text into known sections keyed by section type."""
    heading_to_key = {h: key for key, hs in _SECTION_HEADINGS.items() for h in hs}
    sections: dict[str, list[str]] = {"_header": []}
    current = "_header"
    for line in text.splitlines():
        stripped = line.strip().rstrip(":").strip()
        key = heading_to_key.get(stripped.lower()) if 0 < len(stripped) <= 40 else None
        if key:
            current = key
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def _guess_name(text: str) -> str:
    for line in text.splitlines()[:8]:
        line = line.strip()
        if not line or any(ch in line for ch in "@|:/0123456789"):
            continue
        if re.search(r"resume|curriculum|vitae|\bcv\b", line, re.IGNORECASE):
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(re.fullmatch(r"[A-Za-z][A-Za-z.'\-]*", w) for w in words):
            return line.title() if line.isupper() else line
    return ""


_YEARS_STATED = re.compile(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years|yrs)", re.IGNORECASE)
_DATE_RANGE = re.compile(
    r"((?:19|20)\d{2})\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current|now|till date|date)",
    re.IGNORECASE,
)


def estimate_years(text: str, sections: dict[str, str]) -> float | None:
    """Prefer an explicit 'N+ years' statement; else span job date ranges."""
    stated = [float(m) for m in _YEARS_STATED.findall(text[:3000]) if 0 < float(m) <= 45]
    if stated:
        return max(stated)

    experience = sections.get("experience")
    if not experience:
        return None
    this_year = date.today().year
    intervals: list[tuple[int, int]] = []
    for start, end in _DATE_RANGE.findall(experience):
        s = int(start)
        e = this_year if not end[:1].isdigit() else int(end)
        if s <= e <= this_year:
            intervals.append((s, e))
    if not intervals:
        return None
    # Merge overlapping intervals so concurrent jobs aren't double counted.
    intervals.sort()
    total, cur_s, cur_e = 0, *intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return float(max(total, 0)) or None


def infer_seniority(years: float | None, text: str) -> str:
    head = text[:1500].lower()
    if re.search(r"\b(principal|staff engineer|architect|director|head of)\b", head):
        return "Lead"
    if re.search(r"\b(lead|manager)\b", head) and (years or 0) >= 6:
        return "Lead"
    if years is None:
        return "Senior" if re.search(r"\bsenior\b|\bsr\.?\b", head) else ""
    if years < 1:
        return "Entry"
    if years < 3:
        return "Junior"
    if years < 6:
        return "Mid"
    if years < 10:
        return "Senior"
    return "Lead"


def _detect_locations(text: str) -> list[str]:
    found: list[str] = []
    for city in [*_CITIES, "Madras"]:
        if re.search(rf"\b{city}\b", text, re.IGNORECASE):
            canonical = _CITY_CANONICAL.get(city, city)
            if canonical not in found:
                found.append(canonical)
    return found


def build_search_queries(roles: list[str], skills: list[str], limit: int = 6) -> list[str]:
    queries: list[str] = []
    for role in roles[:3]:
        queries.append(role)
    for role in roles[:2]:
        for skill in skills[:2]:
            queries.append(f"{skill} {role}")
    if not roles:
        queries.extend(f"{skill} developer" for skill in skills[:3])
    deduped = list(dict.fromkeys(q for q in queries if q))
    return deduped[:limit]


def _merge_locations(detected: list[str]) -> list[str]:
    return list(dict.fromkeys([*DEFAULT_PREFERRED_LOCATIONS, *detected]))


def analyze_local(text: str) -> ResumeProfile:
    sections = split_sections(text)
    skills = extract_skills(text)
    years = estimate_years(text, sections)

    # Roles: prefer the header/summary (what they are), then experience (what they've been).
    role_source = "\n".join(sections.get(k, "") for k in ("_header", "summary", "experience")) or text
    roles = extract_roles(role_source)[:4]

    summary = sections.get("summary", "")
    if not summary:
        summary = sections.get("_header", text)[:400]
    summary = re.sub(r"\s+", " ", summary).strip()[:600]

    return ResumeProfile(
        candidate_name=_guess_name(text),
        summary=summary,
        target_roles=roles,
        skills=skills[:40],
        years_of_experience=years,
        seniority=infer_seniority(years, text),
        preferred_locations=_merge_locations(_detect_locations(sections.get("_header", text[:800]))),
        search_queries=build_search_queries(roles, skills),
        raw_text_preview=text[:PREVIEW_CHARS],
        analysis_engine="local",
    )


# --------------------------------------------------------------------------- #
# 3. Gemini analysis
# --------------------------------------------------------------------------- #
class _GeminiResume(BaseModel):
    candidate_name: str = ""
    summary: str = Field(default="", description="2-3 sentence professional summary")
    target_roles: list[str] = Field(default_factory=list, description="Up to 4 job titles this person should apply for")
    skills: list[str] = Field(default_factory=list, description="Concrete technical and domain skills, most important first")
    years_of_experience: float | None = Field(default=None, description="Total professional experience in years")
    seniority: str = Field(default="", description="One of: Entry, Junior, Mid, Senior, Lead")
    current_location: str = ""
    search_queries: list[str] = Field(default_factory=list, description="Up to 6 short job-board search queries")


_GEMINI_PROMPT = """You are a technical recruiter. Extract a structured profile from the resume below.
Rules:
- Only use facts in the resume; leave a field empty if unknown.
- target_roles: realistic next-step job titles (max 4), standard industry titles.
- skills: concrete skills/tools/technologies only (no soft skills), max 40, most relevant first.
- years_of_experience: professional work experience only (exclude education/internships unless that's all there is).
- search_queries: short job-title style queries a job board understands, e.g. "Python Data Engineer" (max 6).
  Do NOT include locations, "remote", "jobs" or company names in queries — location is applied separately.
- The candidate is searching for jobs in Chennai, India and remote roles.

RESUME:
<<<
{resume}
>>>"""


async def analyze_gemini(text: str, api_key: str, model: str) -> _GeminiResume:
    return await generate_json(
        api_key=api_key,
        model=model,
        prompt=_GEMINI_PROMPT.format(resume=text[:GEMINI_MAX_INPUT_CHARS]),
        schema=_GeminiResume,
        temperature=0.1,
    )


def _merge(local: ResumeProfile, ai: _GeminiResume) -> ResumeProfile:
    ai_skills = [normalize_skill(s) for s in ai.skills if s.strip()]
    skills = list(dict.fromkeys([*ai_skills, *local.skills]))[:40]
    roles = [r.strip() for r in ai.target_roles if r.strip()][:4] or local.target_roles
    queries = [q.strip() for q in ai.search_queries if q.strip()][:6] or build_search_queries(roles, skills)
    detected = [ai.current_location] if ai.current_location else []
    seniority = ai.seniority.strip().title()
    if seniority not in {"Entry", "Junior", "Mid", "Senior", "Lead"}:
        seniority = local.seniority

    return ResumeProfile(
        candidate_name=ai.candidate_name.strip() or local.candidate_name,
        summary=ai.summary.strip() or local.summary,
        target_roles=roles,
        skills=skills,
        years_of_experience=ai.years_of_experience if ai.years_of_experience is not None else local.years_of_experience,
        seniority=seniority,
        preferred_locations=list(dict.fromkeys([*local.preferred_locations, *detected])),
        search_queries=queries,
        raw_text_preview=local.raw_text_preview,
        analysis_engine="gemini",
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
async def analyze_resume(
    filename: str,
    data: bytes,
    gemini_api_key: str = "",
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> ResumeProfile:
    """Parse a resume file into a ResumeProfile. Raises ResumeParseError on bad input."""
    return await analyze_text(extract_text(filename, data), gemini_api_key, gemini_model)


async def analyze_text(
    text: str,
    gemini_api_key: str = "",
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> ResumeProfile:
    """Analyse already-extracted resume text (local heuristics, then Gemini if a key is set)."""
    local = analyze_local(text)
    if not gemini_api_key:
        return local
    try:
        ai = await analyze_gemini(text, gemini_api_key, gemini_model)
    except Exception as exc:  # network, quota, bad key, schema drift -> degrade gracefully
        log.warning("Gemini resume analysis failed, using local analysis: %s", exc)
        local.analysis_engine = "local (gemini failed)"
        return local
    return _merge(local, ai)
