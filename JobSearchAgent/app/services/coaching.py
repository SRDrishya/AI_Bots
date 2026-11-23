"""Application kit: tailored cover letter + resume tips + ATS keywords + talking points.

Honesty is the core rule: Gemini may only use facts present in the resume. Gaps
are addressed truthfully (adjacent experience, willingness to learn), never by
claiming skills the candidate doesn't have.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.models import ApplicationKit, KitTone, ResumeProfile, ScoredJob
from app.services.ai import generate_json
from app.services.storage import Storage, now_iso

MAX_CACHED_KITS = 100
RESUME_CHARS = 12_000
JOB_CHARS = 6_000

_TONE = {
    "professional": "confident and professional",
    "warm": "warm and personable while still professional",
    "concise": "brief and direct (about 150 words)",
}


class _KitOut(BaseModel):
    cover_letter: str = Field(description="Plain text, no markdown, paragraphs separated by blank lines")
    resume_tips: list[str] = Field(description="3-6 concrete resume edits for this job")
    keywords: list[str] = Field(description="5-12 terms from the job description the candidate genuinely has")
    talking_points: list[str] = Field(description="3-5 interview talking points, including how to address the main gap")


_PROMPT = """You are an expert career coach helping a candidate in India apply for one specific job.

STRICT HONESTY RULES
- Use ONLY facts found in the candidate's resume below. Never invent employers, projects, numbers,
  certifications, degrees or skills.
- If the job needs something the resume doesn't show (likely gaps: {missing}), do NOT claim it.
  Address it honestly in talking points (adjacent experience, how they'd ramp up). The cover letter may
  briefly mention eagerness to grow in it, or simply focus on strengths.
- Resume tips must rephrase or reorder what the candidate already has, or suggest truthful additions
  (e.g. "if you used X in project Y, say so"). Never suggest adding a skill they lack.
- keywords: only terms from the job description that the resume shows the candidate has.

COVER LETTER
- Tone: {tone}. {length}
- Start with "Dear Hiring Manager," (or the hiring manager's name if the job description gives one).
- Name the actual company and role; no placeholders like [Company] or [Your Name].
- Connect 2-3 of the candidate's most relevant, concrete experiences to the job's key requirements.
- End with a short closing and the candidate's name: {name}.
- Plain text only (no markdown, no bullet points), Indian/British English spelling.
- Use real line breaks: "\\n\\n" between paragraphs, and put the sign-off and the name on separate lines.

JOB
Title: {title}
Company: {company}
Location: {location}
Description:
{description}

CANDIDATE PROFILE
Name: {name}; seniority: {seniority}; experience: {years} years
Target roles: {roles}
Skills: {skills}

CANDIDATE RESUME
<<<
{resume}
>>>"""


_SIGN_OFF = re.compile(r"\b(Sincerely|Yours sincerely|Yours faithfully|Kind regards|Best regards|Warm regards|"
                       r"Regards|Best|Thank you),[ \t]+(?=\S)")
_SALUTATION = re.compile(r"^(Dear [^,\n]{1,60},)[ \t]+(?=\S)")


def _clean_letter(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), text)  # stray markdown bold
    text = text.replace("\r\n", "\n").strip()
    if "\n" not in text:
        # Gemini sometimes separates paragraphs with runs of spaces inside the JSON string.
        text = re.sub(r"[ \t]{2,}", "\n\n", text)
        text = _SALUTATION.sub(r"\1\n\n", text)
    text = _SIGN_OFF.sub(lambda m: f"{m.group(1)},\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _clean_list(items: list[str], limit: int) -> list[str]:
    out = [re.sub(r"^\s*[-*•\d.)]+\s*", "", i).strip() for i in items]
    return list(dict.fromkeys(i for i in out if i))[:limit]


async def generate_kit(
    job: ScoredJob,
    profile: ResumeProfile,
    resume_text: str,
    *,
    api_key: str,
    model: str,
    tone: KitTone = "professional",
) -> ApplicationKit:
    name = profile.candidate_name or "the candidate"
    prompt = _PROMPT.format(
        missing=", ".join(job.missing_skills) or "none identified",
        tone=_TONE[tone],
        length="Keep it to 3 short paragraphs." if tone == "concise" else "Aim for 200-280 words in 3-4 paragraphs.",
        name=name,
        title=job.title,
        company=job.company,
        location=job.location,
        description=job.description[:JOB_CHARS] or "(no description available — rely on the title)",
        seniority=profile.seniority or "unknown",
        years=f"{profile.years_of_experience:g}" if profile.years_of_experience is not None else "unknown",
        roles=", ".join(profile.target_roles) or "n/a",
        skills=", ".join(profile.skills[:40]) or "n/a",
        resume=(resume_text or profile.raw_text_preview or profile.summary)[:RESUME_CHARS],
    )
    out = await generate_json(api_key=api_key, model=model, prompt=prompt, schema=_KitOut, temperature=0.5)
    return ApplicationKit(
        job_key=job.key,
        tone=tone,
        cover_letter=_clean_letter(out.cover_letter),
        resume_tips=_clean_list(out.resume_tips, 6),
        keywords=_clean_list(out.keywords, 12),
        talking_points=_clean_list(out.talking_points, 5),
        created_at=now_iso(),
    )


# --------------------------------------------------------------------------- cache
class KitCache:
    DOC = "application_kits.json"

    def __init__(self, storage: Storage):
        self.storage = storage

    def get(self, key: str, tone: KitTone) -> ApplicationKit | None:
        raw = self.storage.load_doc(self.DOC, {}).get(f"{key}:{tone}")
        return ApplicationKit.model_validate(raw) if raw else None

    def put(self, kit: ApplicationKit) -> None:
        docs = self.storage.load_doc(self.DOC, {})
        docs[f"{kit.job_key}:{kit.tone}"] = kit.model_dump()
        if len(docs) > MAX_CACHED_KITS:  # keep the newest
            docs = dict(sorted(docs.items(), key=lambda kv: kv[1].get("created_at", ""))[-MAX_CACHED_KITS:])
        self.storage.save_doc(self.DOC, docs)
