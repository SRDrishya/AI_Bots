"""Tailored resume: rewrite the saved original resume for one job description.

Flow: prepare (Gemini -> structured ResumeDocument draft) -> user edits the preview ->
generate (render PDF or DOCX, saved under data/tailored_resumes/<job_key>/).

Honesty is the core rule (same as the cover letter): only facts from the original resume.
Gemini is told so, and every draft is also checked locally — skills and figures that don't
appear in the original are listed as warnings in the preview.
"""
from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app.models import (
    KeywordCoverage,
    KeywordHit,
    ResumeDocument,
    ResumeProfile,
    ScoredJob,
    TailoredFile,
    TailoredResume,
)
from app.services.ai import generate_json
from app.services.resume_render import fit_pdf, render_docx
from app.services.skills import extract_skills, normalize_skill
from app.services.storage import Storage, now_iso

RESUME_CHARS = 14_000
JOB_CHARS = 10_000
MAX_KEYWORDS = 20
MAX_DRAFTS = 100
SHORT_JD_CHARS = 600  # below this the UI suggests pasting the full description


class _TailorOut(BaseModel):
    resume: ResumeDocument
    changes: list[str] = Field(description="4-8 short notes on what you changed vs. the original and why")
    gaps: list[str] = Field(description="Job requirements the original resume does NOT show (not added to the resume)")
    jd_keywords: list[str] = Field(description="10-20 most important skills/tools/terms from the job description")


_PROMPT = """You are an expert resume writer for the Indian tech job market. Rewrite the candidate's
ORIGINAL RESUME so it is tailored to the JOB below, as structured data for a one-page ATS-friendly resume.

STRICT HONESTY RULES (most important)
- Use ONLY facts from the original resume. Never invent or upgrade employers, job titles, dates, degrees,
  institutions, certifications, projects, tools, skills, team sizes, numbers or percentages.
- Keep employer names, job titles, dates, degrees and institutions exactly as in the original.
- You MAY: reorder sections, skills and bullets by relevance to the job; reword bullets with stronger
  action verbs; use the job description's terminology for things the candidate demonstrably did
  (e.g. original "built APIs with FastAPI" -> "Designed REST APIs using FastAPI"); merge or drop
  less relevant bullets/projects; write a new summary from facts in the resume.
- Every skill listed must appear in (or be directly named by) the original resume. If the job needs
  something the resume doesn't show, do NOT add it — list it under "gaps" instead.
- Only use numbers/metrics that are in the original resume.

ONE PAGE (strict)
- summary: 2-3 lines (40-60 words), aimed at this role, no first person ("I").
- contact.headline: a short title matching the target role that is truthful for the candidate.
- skills: 3-5 groups, most job-relevant first, max ~10 items per group.
- experience: newest first; 3-5 bullets for the most relevant/recent role, 1-3 for older roles.
  Each bullet one line or two at most (<= 28 words), starting with an action verb.
- projects: at most 3, the most relevant; 1-2 bullets each. Omit if experience is already strong
  and space is short.
- education: keep degree, institution, dates, CGPA/percentage if present.
- certifications / achievements: only the relevant ones, at most 4 each.
- DROP: photo, date of birth, gender, marital status, nationality, father's name, languages known
  (unless the job asks), hobbies, "Declaration", references, career objective.

FORMAT
- Plain text in every field (no markdown, no bullet characters at the start of bullets).
- contact: copy name, email, phone, location (city only) and LinkedIn/GitHub/portfolio URLs exactly.
- Indian/British English spelling, consistent tense (past for previous roles, present for current).
- changes: what you changed and why (for the candidate to review).
- jd_keywords: the important hard skills / tools / qualifications from the job description,
  whether or not the candidate has them.

JOB
Title: {title}
Company: {company}
Location: {location}
Description:
{description}

CANDIDATE PROFILE (from earlier analysis)
Name: {name}; experience: {years} years; skills: {skills}

ORIGINAL RESUME
<<<
{resume}
>>>"""


# --------------------------------------------------------------------------- text helpers
def _norm(text: str) -> str:
    """Lowercase; keep letters/digits and + # . (C++, C#, Node.js); everything else -> single space."""
    text = re.sub(r"[^a-z0-9+#.]+", " ", (text or "").lower())
    text = re.sub(r"\.(?=\s|$)", " ", text)  # sentence-ending dots aren't part of a word
    return f" {re.sub(r'\s+', ' ', text).strip()} "


def mentions(text_norm: str, term: str) -> bool:
    t = _norm(term).strip()
    return bool(t) and f" {t} " in text_norm


def document_text(doc: ResumeDocument) -> str:
    c = doc.contact
    parts = [c.name, c.headline, doc.summary]
    parts += [f"{g.category}: {', '.join(g.items)}" for g in doc.skills]
    for r in doc.experience:
        parts += [r.title, r.company, r.location, r.dates, *r.bullets]
    for p in doc.projects:
        parts += [p.name, p.tech, p.dates, *p.bullets]
    for e in doc.education:
        parts += [e.degree, e.institution, e.location, e.dates, e.details]
    parts += doc.certifications + doc.achievements
    return "\n".join(x for x in parts if x)


def _clean_items(items: list[str], limit: int) -> list[str]:
    out = [re.sub(r"^\s*[-*•▪●•]+\s*", "", i or "").strip() for i in items]
    return list(dict.fromkeys(i for i in out if i))[:limit]


def tidy(doc: ResumeDocument) -> ResumeDocument:
    """Strip stray bullet characters / empties that the AI or the editor may leave behind."""
    doc = doc.model_copy(deep=True)
    doc.contact.links = _clean_items(doc.contact.links, 4)
    for g in doc.skills:
        g.items = _clean_items(g.items, 30)
    doc.skills = [g for g in doc.skills if g.items]
    for entry in [*doc.experience, *doc.projects]:
        entry.bullets = _clean_items(entry.bullets, 12)
    doc.experience = [r for r in doc.experience if r.title.strip() or r.company.strip() or r.bullets]
    doc.projects = [p for p in doc.projects if p.name.strip() or p.bullets]
    doc.education = [e for e in doc.education if e.degree.strip() or e.institution.strip()]
    doc.certifications = _clean_items(doc.certifications, 10)
    doc.achievements = _clean_items(doc.achievements, 10)
    return doc


# --------------------------------------------------------------------------- checks
def keyword_list(jd_keywords: list[str], job_description: str) -> list[str]:
    terms = _clean_items(jd_keywords, MAX_KEYWORDS) + extract_skills(job_description)
    seen, out = set(), []
    for t in terms:
        key = _norm(normalize_skill(t)).strip()
        if key and key not in seen and len(t) <= 40:
            seen.add(key)
            out.append(t)
    return out[:MAX_KEYWORDS]


def coverage(keywords: list[str], original_text: str, doc: ResumeDocument) -> KeywordCoverage:
    new_text = document_text(doc)
    orig, new = _norm(original_text), _norm(new_text)
    orig_skills, new_skills = set(extract_skills(original_text)), set(extract_skills(new_text))

    def has(norm: str, skills: set[str], term: str) -> bool:
        return mentions(norm, term) or normalize_skill(term) in skills

    hits = [KeywordHit(term=k, in_original=has(orig, orig_skills, k), in_tailored=has(new, new_skills, k))
            for k in keywords]
    return KeywordCoverage(before=sum(h.in_original for h in hits), after=sum(h.in_tailored for h in hits),
                           total=len(hits), keywords=hits)


_NUMBER = re.compile(r"(?<![\w.])\d[\d,.]*\s*(?:%|\+|x\b|k\b|lakh|crore|cr\b|million|users|clients)?", re.I)


def honesty_warnings(original_text: str, doc: ResumeDocument) -> list[str]:
    """Skills and figures in the tailored resume that the original doesn't contain."""
    orig = _norm(original_text)
    orig_skills = {normalize_skill(s).lower() for s in extract_skills(original_text)}
    warnings = []
    unverified = [item for g in doc.skills for item in g.items
                  if not mentions(orig, item) and normalize_skill(item).lower() not in orig_skills]
    if unverified:
        warnings.append("Skills not found in your original resume — keep only if true: " + ", ".join(unverified))

    digits_orig = re.sub(r"\D", " ", original_text)
    orig_numbers = set(digits_orig.split())
    new_figures = []
    for text in [doc.summary, *(b for r in doc.experience for b in r.bullets),
                 *(b for p in doc.projects for b in p.bullets), *doc.achievements]:
        for m in _NUMBER.finditer(text or ""):
            nums = re.sub(r"\D", " ", m.group()).split()
            if nums and any(n not in orig_numbers for n in nums):
                new_figures.append(m.group().strip())
    if new_figures:
        warnings.append("Figures not in your original resume — check they're accurate: "
                        + ", ".join(dict.fromkeys(new_figures)))
    return warnings


def refresh(draft: TailoredResume, original_text: str) -> TailoredResume:
    """Recompute coverage, honesty warnings and one-page fit after the resume content changed."""
    doc = tidy(draft.resume)
    _, fit = fit_pdf(doc)
    return draft.model_copy(update={
        "resume": doc,
        "coverage": coverage(draft.jd_keywords, original_text, doc),
        "warnings": honesty_warnings(original_text, doc),
        "fit": fit,
        "updated_at": now_iso(),
    })


# --------------------------------------------------------------------------- AI
async def prepare_tailored(
    job: ScoredJob,
    job_description: str,
    profile: ResumeProfile,
    resume_text: str,
    *,
    api_key: str,
    model: str,
) -> TailoredResume:
    prompt = _PROMPT.format(
        title=job.title, company=job.company, location=job.location,
        description=job_description[:JOB_CHARS] or "(no description available — tailor to the job title)",
        name=profile.candidate_name or "unknown",
        years=f"{profile.years_of_experience:g}" if profile.years_of_experience is not None else "unknown",
        skills=", ".join(profile.skills[:40]) or "n/a",
        resume=resume_text[:RESUME_CHARS],
    )
    out = await generate_json(api_key=api_key, model=model, prompt=prompt, schema=_TailorOut, temperature=0.3)
    now = now_iso()
    draft = TailoredResume(
        job_key=job.key, job_title=job.title, company=job.company, job_description=job_description,
        resume=out.resume, changes=_clean_items(out.changes, 10), gaps=_clean_items(out.gaps, 10),
        jd_keywords=keyword_list(out.jd_keywords, job_description), created_at=now,
    )
    return refresh(draft, resume_text)


# --------------------------------------------------------------------------- files
def file_stem(company: str, title: str) -> str:
    """CompanyName_Title_Resume (filesystem- and email-safe)."""
    def part(s: str, n: int) -> str:
        return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", s or "")).strip("_")[:n]
    return "_".join(p for p in (part(company, 40), part(title, 60), "Resume") if p)


def render(draft: TailoredResume, fmt: str) -> tuple[bytes, str]:
    pdf_bytes, fit = fit_pdf(draft.resume)
    if fmt == "pdf":
        return pdf_bytes, "application/pdf"
    return (render_docx(draft.resume, fit.font_size),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document")


class TailoredStore:
    """Drafts in data/tailored_resumes.json; generated files in data/tailored_resumes/<job_key>/."""
    DOC = "tailored_resumes.json"
    DIR = "tailored_resumes"

    def __init__(self, storage: Storage):
        self.storage = storage

    @staticmethod
    def valid_key(key: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key or ""))

    def get(self, key: str) -> TailoredResume | None:
        raw = self.storage.load_doc(self.DOC, {}).get(key)
        if not raw:
            return None
        try:
            return TailoredResume.model_validate(raw)
        except ValidationError:
            return None

    def put(self, draft: TailoredResume) -> None:
        docs = self.storage.load_doc(self.DOC, {})
        docs[draft.job_key] = draft.model_dump()
        if len(docs) > MAX_DRAFTS:  # keep the most recently edited
            docs = dict(sorted(docs.items(), key=lambda kv: kv[1].get("updated_at", ""))[-MAX_DRAFTS:])
        self.storage.save_doc(self.DOC, docs)

    def list(self) -> list[TailoredResume]:
        out = []
        for raw in self.storage.load_doc(self.DOC, {}).values():
            try:
                out.append(TailoredResume.model_validate(raw))
            except ValidationError:
                continue
        return sorted(out, key=lambda d: d.updated_at, reverse=True)

    def folder(self, key: str) -> Path:
        return self.storage.root / self.DIR / key

    def save_file(self, draft: TailoredResume, fmt: str, data: bytes) -> TailoredResume:
        filename = f"{file_stem(draft.company, draft.job_title)}.{fmt}"
        folder = self.folder(draft.job_key)
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f".{filename}.tmp"
        tmp.write_bytes(data)
        tmp.replace(folder / filename)
        files = [f for f in draft.files if f.filename != filename]
        files.append(TailoredFile(format=fmt, filename=filename, created_at=now_iso()))
        draft = draft.model_copy(update={"files": files})
        self.put(draft)
        return draft

    def file_path(self, key: str, filename: str) -> Path | None:
        draft = self.get(key)
        if draft is None or filename not in {f.filename for f in draft.files}:
            return None
        path = self.folder(key) / filename
        return path if path.is_file() else None
