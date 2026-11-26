"""FastAPI app: JSON API under /api plus the single-page UI at /.

Every bank is one technology at one experience level: routes take the technology slug in the path
and the experience key as ?exp= (default 5-7 years).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from app.config import (
    BASE_DIR,
    DEFAULT_EXPERIENCE,
    EXPERIENCE_LEVELS,
    Experience,
    get_env_settings,
    get_experience,
)
from app.models import (
    AddTechnologyRequest,
    Attempt,
    Bank,
    EvaluateRequest,
    ExportRequest,
    PreferencesUpdate,
    ProgressUpdate,
    ResumeUpdate,
    Technology,
    bank_key,
)
from app.services import ai, catalog, claude_code, export, practice
from app.services.ai import NO_KEY_HINT, AIError
from app.services.builder import BankBuilder, regenerate_answer
from app.services.storage import Storage, now_iso
from app.services.vectors import EmbeddingError, QuestionIndex

log = logging.getLogger(__name__)
STATIC_DIR = BASE_DIR / "static"
BUILD_MODES = ("build", "more", "answers")
EXPORT_TYPES = {"pdf": "application/pdf", "html": "text/html; charset=utf-8",
                "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = get_env_settings().data_dir
    storage = Storage(data_dir)
    index = QuestionIndex(data_dir / "chroma")
    app.state.storage = storage
    app.state.index = index
    app.state.builder = BankBuilder(storage, index)
    storage.load_resume()  # imports the JobSearchAgent resume on first start
    ai.set_mode(storage.load_provider())
    yield


app = FastAPI(title="InterviewPrep", lifespan=lifespan)


@app.exception_handler(AIError)
async def ai_error_handler(_: Request, exc: AIError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


def _storage(request: Request) -> Storage:
    return request.app.state.storage


def _builder(request: Request) -> BankBuilder:
    return request.app.state.builder


def _index(request: Request) -> QuestionIndex:
    return request.app.state.index


def _technologies(storage: Storage) -> list[Technology]:
    return catalog.merge(storage.load_custom_technologies())


def _experience(exp: str) -> Experience:
    experience = get_experience(exp)
    if experience is None:
        raise HTTPException(400, f"Unknown experience level '{exp}'")
    return experience


def _resolve(request: Request, slug: str, exp: str) -> tuple[Technology, Experience, str]:
    tech = catalog.find(_technologies(_storage(request)), slug)
    if tech is None:
        raise HTTPException(404, "Unknown technology")
    experience = _experience(exp)
    return tech, experience, bank_key(slug, experience.key)


def _bank_or_404(storage: Storage, key: str) -> Bank:
    bank = storage.load_bank(key)
    if bank is None:
        raise HTTPException(404, "No questions yet for this technology and experience — build the bank first")
    return bank


def _question_or_404(bank: Bank, question_id: str):
    question = next((q for q in bank.questions if q.id == question_id), None)
    if question is None:
        raise HTTPException(404, "Unknown question")
    return question


def _require_ai() -> None:
    if not ai.ai_available():
        raise HTTPException(400, f"No AI provider configured — add {NO_KEY_HINT} and restart")


def _not_running(request: Request, key: str) -> None:
    if _builder(request).is_running(key):
        raise HTTPException(409, "Wait for the running build to finish")


Exp = Query(DEFAULT_EXPERIENCE, alias="exp")


# ---------- UI ----------

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# ---------- status & resume ----------

@app.get("/api/status")
async def status(request: Request):
    env = get_env_settings()
    return {
        "ai": ai.describe(), "ai_available": ai.ai_available(), "no_key_hint": NO_KEY_HINT,
        "semantic": _index(request).enabled, "embedding_model": env.embedding_model,
        "resume_chars": len(_storage(request).load_resume()), "questions_per_tech": env.questions_per_tech,
        "experiences": [{"key": e.key, "label": e.label} for e in EXPERIENCE_LEVELS],
        "default_experience": DEFAULT_EXPERIENCE,
        "companies": _storage(request).load_companies(),
        "provider": ai.mode(),
        "providers": [{"key": k, "label": v} for k, v in ai.MODES.items()],
        "claude_code_path": claude_code.find_executable(),
    }


@app.get("/api/resume")
async def get_resume(request: Request):
    return {"text": _storage(request).load_resume()}


@app.put("/api/resume")
async def put_resume(body: ResumeUpdate, request: Request):
    _storage(request).save_resume(body.text)
    return {"ok": True, "chars": len(body.text.strip())}


@app.put("/api/preferences")
async def put_preferences(body: PreferencesUpdate, request: Request):
    """Target companies (their candidates' interview reports are searched first) and the AI provider."""
    storage = _storage(request)
    if body.provider is not None:
        if body.provider not in ai.MODES:
            raise HTTPException(400, f"provider must be one of {', '.join(ai.MODES)}")
        if body.provider == "claude_code" and not claude_code.find_executable():
            raise HTTPException(400, "Claude Code was not found — install it, or set CLAUDE_CODE_PATH in .env")
        storage.save_provider(body.provider)
        ai.set_mode(body.provider)
    if body.companies is not None:
        storage.save_companies(body.companies)
    return {"companies": storage.load_companies(), "provider": ai.mode()}


# ---------- technologies ----------

@app.get("/api/technologies")
async def list_technologies(request: Request, exp: str = Exp):
    experience = _experience(exp)
    storage, builder = _storage(request), _builder(request)
    counts = storage.bank_counts()
    out = []
    for tech in _technologies(storage):
        key = bank_key(tech.slug, experience.key)
        out.append({**tech.model_dump(), "count": counts.get(key, 0), "job": builder.status(key).model_dump()})
    return out


@app.post("/api/technologies")
async def add_technology(body: AddTechnologyRequest, request: Request):
    storage = _storage(request)
    name = body.name.strip()
    existing = next((t for t in _technologies(storage)
                     if catalog.same_name(t.name, name) or t.slug == catalog.slugify(name)), None)
    if existing:
        return existing
    tech = Technology(slug=catalog.slugify(name), name=name, kind=body.kind, focus=body.focus.strip(),
                      group="Added by you")
    storage.save_custom_technologies([*storage.load_custom_technologies(), tech])
    return tech


@app.delete("/api/technologies/{slug}")
async def delete_technology(slug: str, request: Request):
    storage = _storage(request)
    tech = catalog.find(_technologies(storage), slug)
    if tech is None:
        raise HTTPException(404, "Unknown technology")
    if tech.builtin:
        raise HTTPException(400, "Built-in technologies can't be removed (you can delete their questions)")
    for e in EXPERIENCE_LEVELS:
        key = bank_key(slug, e.key)
        _not_running(request, key)
        await _drop_bank(request, key)
    storage.save_custom_technologies([t for t in storage.load_custom_technologies() if t.slug != slug])
    return {"ok": True}


async def _drop_bank(request: Request, key: str) -> bool:
    deleted = _storage(request).delete_bank(key)
    if _index(request).enabled:
        try:
            await _index(request).delete_bank(key)
        except EmbeddingError:
            pass
    return deleted


# ---------- banks ----------

@app.get("/api/banks/{slug}")
async def get_bank(slug: str, request: Request, exp: str = Exp):
    _, _, key = _resolve(request, slug, exp)
    storage = _storage(request)
    bank = storage.load_bank(key)
    progress = storage.load_progress()
    ids = {q.id for q in bank.questions} if bank else set()
    return {
        "bank": bank.model_dump() if bank else None,
        "progress": {k: v.model_dump() for k, v in progress.items() if k in ids},
        "job": _builder(request).status(key).model_dump(),
    }


@app.post("/api/banks/{slug}/build")
async def build_bank(slug: str, request: Request, exp: str = Exp, mode: str = Query("build")):
    if mode not in BUILD_MODES:
        raise HTTPException(400, f"mode must be one of {', '.join(BUILD_MODES)}")
    _require_ai()
    tech, experience, key = _resolve(request, slug, exp)
    if mode != "build":
        _bank_or_404(_storage(request), key)
    return _builder(request).start(tech, experience, mode)


@app.get("/api/jobs/{slug}")
async def job_status(slug: str, request: Request, exp: str = Exp):
    _, _, key = _resolve(request, slug, exp)
    return _builder(request).status(key)


@app.delete("/api/banks/{slug}")
async def delete_bank(slug: str, request: Request, exp: str = Exp):
    _, _, key = _resolve(request, slug, exp)
    _not_running(request, key)
    return {"deleted": await _drop_bank(request, key)}


@app.post("/api/banks/{slug}/questions/{question_id}/answer")
async def regenerate(slug: str, question_id: str, request: Request, exp: str = Exp):
    _require_ai()
    tech, experience, key = _resolve(request, slug, exp)
    _not_running(request, key)
    storage = _storage(request)
    bank = _bank_or_404(storage, key)
    question = _question_or_404(bank, question_id)
    await regenerate_answer(tech, experience, question, storage.load_resume())
    storage.save_bank(bank)
    return question


@app.delete("/api/banks/{slug}/questions/{question_id}")
async def delete_question(slug: str, question_id: str, request: Request, exp: str = Exp):
    _, _, key = _resolve(request, slug, exp)
    _not_running(request, key)
    storage = _storage(request)
    bank = _bank_or_404(storage, key)
    _question_or_404(bank, question_id)
    bank.questions = [q for q in bank.questions if q.id != question_id]
    storage.save_bank(bank)
    if _index(request).enabled:
        try:
            await _index(request).delete([question_id])
        except EmbeddingError:
            pass
    return {"ok": True}


# ---------- semantic search ----------

@app.get("/api/banks/{slug}/search")
async def search(slug: str, request: Request, q: str = Query(min_length=2, max_length=300), exp: str = Exp):
    """Questions ranked by meaning (LangChain + Chroma); falls back to word matching without embeddings."""
    _, _, key = _resolve(request, slug, exp)
    bank = _bank_or_404(_storage(request), key)
    index = _index(request)
    if index.enabled:
        try:
            await index.add(key, bank.questions)  # no-op when already indexed
            hits = await index.search(key, q, k=min(40, len(bank.questions)))
            return {"mode": "semantic", "results": [{"id": i, "score": s} for i, s in hits if s >= 0.35]}
        except EmbeddingError as exc:
            log.warning("Semantic search failed, using word match: %s", exc)
    term = q.lower()
    return {"mode": "text", "results": [{"id": x.id, "score": 1.0} for x in bank.questions
                                        if term in x.question.lower()]}


@app.get("/api/banks/{slug}/questions/{question_id}/related")
async def related(slug: str, question_id: str, request: Request, exp: str = Exp):
    _, _, key = _resolve(request, slug, exp)
    bank = _bank_or_404(_storage(request), key)
    _question_or_404(bank, question_id)
    index = _index(request)
    if not index.enabled:
        return {"results": [], "note": "Related questions need a Gemini key (embeddings)"}
    try:
        await index.add(key, bank.questions)
        hits = await index.related(key, question_id, k=5)
    except EmbeddingError as exc:
        return {"results": [], "note": str(exc)}
    return {"results": [{"id": i, "score": s} for i, s in hits if s >= 0.5]}


# ---------- export (after the user has reviewed and selected questions) ----------

@app.post("/api/banks/{slug}/export")
async def export_bank(slug: str, body: ExportRequest, request: Request, exp: str = Exp):
    tech, experience, key = _resolve(request, slug, exp)
    bank = _bank_or_404(_storage(request), key)
    wanted = set(body.question_ids)
    questions = [q for q in bank.questions if q.id in wanted]
    if not questions:
        raise HTTPException(400, "None of the selected questions are in this bank")
    opts = export.ExportOptions(
        title=f"{tech.name} — Interview Questions & Answers",
        subtitle=export.subtitle(len(questions), experience.label),
        include_sources=body.include_sources, include_angle=body.include_angle)
    if body.format == "pdf":
        content: bytes | str = export.to_pdf(bank, questions, opts)
    elif body.format == "docx":
        content = export.to_docx(bank, questions, opts)
    else:
        content = export.to_html(bank, questions, opts)
    filename = f"{slug}-{experience.key}-interview-questions.{body.format}"
    return Response(content, media_type=EXPORT_TYPES[body.format],
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# ---------- practice ----------

@app.put("/api/progress/{question_id}")
async def set_progress(question_id: str, body: ProgressUpdate, request: Request):
    def update(entry):
        entry.status = body.status
    return _storage(request).update_progress(question_id, update)


@app.post("/api/practice/evaluate")
async def evaluate_answer(body: EvaluateRequest, request: Request):
    _require_ai()
    tech, experience, key = _resolve(request, body.slug, body.experience)
    storage = _storage(request)
    question = _question_or_404(_bank_or_404(storage, key), body.question_id)
    evaluation = await practice.evaluate(tech.name, experience, question, body.answer, storage.load_resume())

    def update(entry):
        entry.attempts = [*entry.attempts, Attempt(answer=body.answer, evaluation=evaluation, at=now_iso())][-10:]
        if entry.status == "new":
            entry.status = "known" if evaluation.score >= 8 else "review"
    progress = storage.update_progress(question.id, update)
    return {"evaluation": evaluation, "progress": progress}
