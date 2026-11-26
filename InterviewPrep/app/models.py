"""Pydantic models: stored data, API payloads, and the JSON shapes asked of the AI."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["beginner", "intermediate", "advanced"]
LEVELS: tuple[str, ...] = ("beginner", "intermediate", "advanced")
# tech: a language/framework/database/tool. design: architecture & patterns.
# behavioral: leadership / HR rounds. ai: LLM & RAG. role: a whole job interview (mixes several skills).
Kind = Literal["tech", "design", "behavioral", "ai", "role"]
Status = Literal["new", "known", "review"]
ExportFormat = Literal["pdf", "docx", "html"]


def normalize_level(value: str) -> str:
    value = (value or "").strip().lower()
    if value.startswith(("beg", "basic", "easy", "jun", "fresh")):
        return "beginner"
    if value.startswith(("adv", "hard", "expert", "sen", "lead", "arch")):
        return "advanced"
    return "intermediate"


def bank_key(slug: str, experience: str) -> str:
    return f"{slug}--{experience}"


# ---------- stored data ----------

class Source(BaseModel):
    title: str
    url: str


class Answer(BaseModel):
    summary: str = ""                 # the 1-2 sentence answer to say first
    explanation: str = ""             # markdown, the detailed answer
    code_language: str = ""
    code_example: str = ""
    key_points: list[str] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
    your_angle: str = ""              # how to tie the answer to the candidate's own experience
    generated_at: str = ""


class Question(BaseModel):
    id: str
    question: str                     # as written on the source page
    category: str
    level: Level = "intermediate"
    sources: list[Source] = Field(default_factory=list)   # the page(s) the question was found on
    fits_experience: bool = True      # False: real question, but basic for the selected experience level
    reported: bool = False            # a candidate reported being asked this in a real interview
    companies: list[str] = Field(default_factory=list)    # companies it was reported at
    likelihood: float = 0.0           # how likely it is to be asked (see builder.likelihood)
    origin: Literal["web", "ai"] = "web"   # ai: written by Claude Code (no web source)
    commonness: int = 0               # ai questions: how often it comes up, 1-5 (the AI's estimate)
    answer: Answer | None = None
    created_at: str = ""


class Technology(BaseModel):
    slug: str
    name: str
    kind: Kind = "tech"
    group: str = "Custom"
    focus: str = ""        # extra context for the AI, e.g. "versions 8 and 9"
    search_terms: str = ""  # what to type into a search engine (defaults to name)
    ambitionbox_roles: list[str] = Field(default_factory=list)  # AmbitionBox role slugs read directly
    builtin: bool = False


class Bank(BaseModel):
    key: str
    slug: str
    name: str
    kind: Kind = "tech"
    experience: str
    categories: list[str] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    used_urls: list[str] = Field(default_factory=list)    # pages already read (skipped next time)
    search_rounds: int = 0                                 # query sets already used
    updated_at: str = ""


class Evaluation(BaseModel):
    score: int = 0          # 0-10
    verdict: str = ""
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    improved_answer: str = ""


class Attempt(BaseModel):
    answer: str
    evaluation: Evaluation
    at: str


class QuestionProgress(BaseModel):
    status: Status = "new"
    attempts: list[Attempt] = Field(default_factory=list)
    updated_at: str = ""


class JobStatus(BaseModel):
    key: str
    mode: str = ""
    state: Literal["idle", "running", "done", "error"] = "idle"
    stage: str = ""
    done: int = 0
    total: int = 0
    message: str = ""
    started_at: str = ""
    finished_at: str = ""


# ---------- API payloads ----------

class AddTechnologyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    kind: Kind = "tech"
    focus: str = Field(default="", max_length=300)


class ProgressUpdate(BaseModel):
    status: Status


class EvaluateRequest(BaseModel):
    slug: str
    experience: str
    question_id: str
    answer: str = Field(min_length=1, max_length=20_000)


class ExportRequest(BaseModel):
    format: ExportFormat
    question_ids: list[str] = Field(min_length=1)
    include_sources: bool = True
    include_angle: bool = True


class ResumeUpdate(BaseModel):
    text: str = Field(max_length=100_000)


class PreferencesUpdate(BaseModel):
    companies: list[str] | None = Field(default=None, max_length=60)
    provider: str | None = None       # auto | gemini | anthropic | claude_code


# ---------- shapes asked of the AI (no defaults: Gemini schemas reject them) ----------

class PlannedCategory(BaseModel):
    name: str
    focus: str


class CategoryPlan(BaseModel):
    categories: list[PlannedCategory]


class ExtractedQuestion(BaseModel):
    question: str
    category: str
    level: str
    fits_experience: bool
    reported_by_candidate: bool
    companies: list[str]


class ExtractedQuestions(BaseModel):
    questions: list[ExtractedQuestion]


class GeneratedQuestion(BaseModel):
    question: str
    category: str
    level: str
    commonly_asked: int


class GeneratedQuestions(BaseModel):
    questions: list[GeneratedQuestion]


class AnswerDraft(BaseModel):
    number: int
    summary: str
    explanation: str
    code_language: str
    code_example: str
    key_points: list[str]
    pitfalls: list[str]
    follow_ups: list[str]
    your_angle: str


class AnswerBatch(BaseModel):
    answers: list[AnswerDraft]


class EvaluationDraft(BaseModel):
    score: int
    verdict: str
    strengths: list[str]
    gaps: list[str]
    improved_answer: str


class SameQuestionVerdict(BaseModel):
    pair: int
    same: bool


class SameQuestionVerdicts(BaseModel):
    verdicts: list[SameQuestionVerdict]
