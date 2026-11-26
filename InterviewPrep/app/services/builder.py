"""Builds a question bank for one technology at one experience level.

Questions only ever come from real web pages; the AI writes the answers.

  1. plan      AI names ~10 categories used to group the questions (it does not write questions)
  2. search    web-grounded AI search finds pages with interview questions for this experience level
  3. read      pages are downloaded and split into chunks (LangChain text splitter)
  4. extract   AI copies the questions out of each chunk; every question is checked against the page
               text (gather.appears_in) and dropped if it isn't there. Questions that fit the experience
               level are kept first; basic ones only fill remaining space and are marked as such.
               Each question keeps the link of the page it came from.
  5. dedupe    exact/word-level duplicates, then semantic duplicates via the LangChain + Chroma index
               (borderline pairs are confirmed by the AI)
  6. answer    detailed answers pitched at the experience level, written in small batches

Steps 2-5 repeat with new search queries until the target is reached (or pages run out).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from urllib.parse import urlparse

import numpy as np

from app.config import (
    AI_CONCURRENCY,
    ANSWER_BATCH_SIZE,
    MAX_PAGES_PER_ROUND,
    MAX_SEARCH_ROUNDS,
    Experience,
    get_env_settings,
)
from app.models import (
    Answer,
    AnswerBatch,
    Bank,
    CategoryPlan,
    ExtractedQuestions,
    GeneratedQuestions,
    JobStatus,
    Question,
    SameQuestionVerdicts,
    Source,
    Technology,
    bank_key,
    normalize_level,
)
from app.services import ai, gather, prompts
from app.services.ai import AIError
from app.services.storage import Storage, now_iso
from app.services.vectors import EmbeddingError, QuestionIndex

log = logging.getLogger(__name__)

MORE_QUESTIONS = 25
CATEGORIES_PER_CALL = 3        # Claude Code mode: categories written per call
CLAUDE_CODE_ANSWER_BATCH = 8   # Claude Code mode: answers per call (each call has a fixed start-up cost)
MAX_SOURCES = 8               # source links kept per question
MAX_PER_PAGE = 6              # new questions one page contributes before other pages get their turn
# Sites made of candidates' own interview reports (not prep articles)
REPORT_SITES = ("ambitionbox.com", "glassdoor.com", "glassdoor.co.in", "leetcode.com", "reddit.com",
                "geeksforgeeks.org")
MAX_OVERSHOOT = 1.2           # keep at most 20% more questions than the target in one build
WORD_DUPLICATE = 0.75         # share of key words two questions must share to be word-level duplicates
AUTO_DUPLICATE = 0.06         # cosine distance: below this two questions are the same, no need to ask
JUDGE_BELOW = 0.14            # between AUTO_DUPLICATE and this, the AI decides (e.g. IEnumerable vs ICollection)
JUDGE_BATCH = 25


# ---------- pure helpers (unit-tested) ----------

def question_id(key: str, text: str) -> str:
    norm = " ".join(re.findall(r"[a-z0-9#+]+", text.lower()))
    return f"{key}-{hashlib.sha1(norm.encode()).hexdigest()[:10]}"


Merge = tuple[Question, Question]   # (duplicate found on another page, the question it repeats)


def _word_match(words: set[str], pool: list[tuple[set[str], Question]]) -> Question | None:
    for other_words, other in pool:
        union = words | other_words
        if not union or len(words & other_words) / len(union) >= WORD_DUPLICATE:
            return other
    return None


def dedupe_words(existing: list[Question], new: list[Question]) -> tuple[list[Question], list[Merge]]:
    """-> (new questions that aren't word-level duplicates, [(duplicate, question it repeats)])."""
    pool = [(gather.key_words(q.question), q) for q in existing]
    kept: list[Question] = []
    merges: list[Merge] = []
    for q in new:
        words = gather.key_words(q.question)
        if len(q.question) < 12 or not words:
            continue
        match = _word_match(words, pool)
        if match is not None:
            merges.append((q, match))
            continue
        pool.append((words, q))
        kept.append(q)
    return kept, merges


def _host(url: str) -> str:
    host = urlparse(url).hostname or url
    return host.removeprefix("www.").removeprefix("old.")


def merge_into(target: Question, duplicate: Question) -> None:
    """The same question was found again: keep one copy, remember every site and company that has it."""
    known = {s.url for s in target.sources}
    for source in duplicate.sources:
        if source.url not in known and len(target.sources) < MAX_SOURCES:
            target.sources.append(source)
            known.add(source.url)
    target.reported = target.reported or duplicate.reported
    target.fits_experience = target.fits_experience or duplicate.fits_experience
    for company in duplicate.companies:
        if company.lower() not in {c.lower() for c in target.companies}:
            target.companies.append(company)


def likelihood(q: Question) -> float:
    """How likely the question is to come up: reported by real candidates (more so on sites made of
    candidate reports), at named companies, on many different sites, right for the experience level."""
    hosts = {_host(s.url) for s in q.sources}
    from_report_site = any(h == site or h.endswith("." + site) for h in hosts for site in REPORT_SITES)
    return round((3.0 if q.reported else 0.0) + (1.0 if from_report_site else 0.0) + min(len(q.companies), 3)
                 + 1.5 * min(max(len(hosts) - 1, 0), 4) + (1.0 if q.fits_experience else 0.0)
                 + q.commonness, 2)  # commonness: AI-generated questions only (the AI's 1-5 estimate)


def spread_by_page(questions: list[Question], per_page: int) -> list[Question]:
    """Reorder so every page gets its first `per_page` questions in before any page gets more,
    keeping the bank from being dominated by one long list. Order within a page is kept."""
    first, overflow, taken = [], [], {}
    for q in questions:
        page = q.sources[0].url if q.sources else ""
        taken[page] = taken.get(page, 0) + 1
        (first if taken[page] <= per_page else overflow).append(q)
    return first + overflow


def clean_question(text: str) -> str:
    text = re.sub(r"^\s*(?:Q(?:uestion)?\s*)?\d+\s*[.):\-]\s*", "", text.strip(), flags=re.IGNORECASE)  # "Q12." / "3)"
    text = " ".join(text.split()).strip(" *-")
    return text[0].upper() + text[1:] if text else text


def match_category(name: str, categories: list[str]) -> str:
    wanted = name.strip().lower()
    for c in categories:
        if c.lower() == wanted:
            return c
    for c in categories:
        if wanted and (wanted in c.lower() or c.lower() in wanted):
            return c
    return categories[-1] if categories else "General"


def category_count(target: int) -> int:
    return max(6, min(12, round(target / 10)))


def filter_questions(questions: list[Question], progress: dict, *, category: str = "", level: str = "",
                     status: str = "", search: str = "") -> list[Question]:
    term = search.strip().lower()
    out = []
    for q in questions:
        if category and q.category != category:
            continue
        if level and q.level != level:
            continue
        if status:
            entry = progress.get(q.id)
            if (entry.status if entry else "new") != status:
                continue
        if term and term not in q.question.lower():
            continue
        out.append(q)
    return out


# ---------- AI steps ----------

async def plan_categories(tech: Technology, experience: Experience, target: int) -> list[str]:
    count = category_count(target)
    result = await ai.generate_json(prompt=prompts.plan(tech, count, experience), schema=CategoryPlan, fast=True)
    names = list(dict.fromkeys(c.name.strip() for c in result.categories if c.name.strip()))
    if not names:
        raise AIError("The AI returned no categories")
    return [*names[:count], "Other"]


async def extract_from_chunk(tech: Technology, experience: Experience, key: str, categories: list[str],
                             doc, stats: dict) -> list[Question]:
    url, title = doc.metadata["url"], doc.metadata["title"]
    result = await ai.generate_json(prompt=prompts.extract(tech, experience, categories, doc.page_content, url),
                                    schema=ExtractedQuestions, fast=True)
    created = now_iso()
    out = []
    for item in result.questions:
        text = clean_question(item.question)
        if not gather.appears_in(text, doc.page_content):
            stats["not_on_page"] += 1
            continue
        if not item.fits_experience:
            stats["off_level"] += 1
        out.append(Question(id=question_id(key, text), question=text,
                            category=match_category(item.category, categories),
                            level=normalize_level(item.level), sources=[Source(title=title, url=url)],
                            fits_experience=item.fits_experience, reported=item.reported_by_candidate,
                            companies=[c.strip() for c in item.companies if c.strip()][:5], created_at=created))
    return out


async def judge_duplicates(pairs: list[tuple[Question, str]]) -> set[str]:
    """Ids of candidates the AI says ask the same thing as their nearest existing question."""
    duplicates: set[str] = set()
    for start in range(0, len(pairs), JUDGE_BATCH):
        batch = pairs[start:start + JUDGE_BATCH]
        numbered = [(i + 1, cand.question, other) for i, (cand, other) in enumerate(batch)]
        result = await ai.generate_json(prompt=prompts.same_questions(numbered), schema=SameQuestionVerdicts,
                                        temperature=0.0, fast=True)
        for verdict in result.verdicts:
            if verdict.same and 1 <= verdict.pair <= len(batch):
                duplicates.add(batch[verdict.pair - 1][0].id)
    return duplicates


async def semantic_dedupe(index: QuestionIndex | None, key: str, existing: list[Question],
                          candidates: list[Question], warnings: list[str]) -> tuple[list[Question], list[Merge]]:
    """-> (genuinely new questions, [(duplicate, question it repeats)]). Duplicates are not thrown away:
    the caller merges their source links and companies into the question they repeat."""
    candidates, merges = dedupe_words(existing, candidates)
    if not candidates or index is None or not index.enabled:
        return candidates, merges
    try:
        await index.add(key, existing)  # make sure everything already in the bank can be compared against
        vectors = np.array(await index.embed([c.question for c in candidates]), dtype=float)
        nearest = await index.nearest(key, vectors.tolist())
    except EmbeddingError as exc:
        warnings.append(f"Semantic duplicate check skipped: {exc}")
        return candidates, merges
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    by_id = {q.id: q for q in existing}

    accepted: list[int] = []
    borderline: list[tuple[Question, str, Question]] = []
    for i, cand in enumerate(candidates):
        best_dist, best = (nearest[i][2], by_id.get(nearest[i][0])) if nearest[i] else (2.0, None)
        if accepted:
            dists = 1.0 - vectors[accepted] @ vectors[i]
            j = int(np.argmin(dists))
            if dists[j] < best_dist:
                best_dist, best = float(dists[j]), candidates[accepted[j]]
        if best is not None and best_dist <= AUTO_DUPLICATE:
            merges.append((cand, best))
            continue
        accepted.append(i)
        if best is not None and best_dist < JUDGE_BELOW:
            borderline.append((cand, best.question, best))

    kept = [candidates[i] for i in accepted]
    if borderline:
        try:
            dupes = await judge_duplicates([(cand, text) for cand, text, _ in borderline])
            merges.extend((cand, target) for cand, _, target in borderline if cand.id in dupes)
            kept = [c for c in kept if c.id not in dupes]
        except AIError as exc:
            warnings.append(f"Could not double-check {len(borderline)} possible duplicates: {exc}")
    return kept, merges


async def answer_batch(tech: Technology, experience: Experience, questions: list[Question], resume: str) -> int:
    """Fill `answer` on each question in place. Returns how many got an answer."""
    numbered = [(i + 1, q.question) for i, q in enumerate(questions)]
    result = await ai.generate_json(prompt=prompts.answers(tech, experience, numbered, resume),
                                    schema=AnswerBatch, temperature=0.3)
    created = now_iso()
    filled = 0
    for draft in result.answers:
        if 1 <= draft.number <= len(questions):
            questions[draft.number - 1].answer = Answer(
                summary=draft.summary.strip(),
                explanation=draft.explanation.strip(),
                code_language=draft.code_language.strip(),
                code_example=_strip_fences(draft.code_example),
                key_points=[p for p in draft.key_points if p.strip()],
                pitfalls=[p for p in draft.pitfalls if p.strip()],
                follow_ups=[p for p in draft.follow_ups if p.strip()],
                your_angle=draft.your_angle.strip(),
                generated_at=created,
            )
            filled += 1
    return filled


def _strip_fences(code: str) -> str:
    code = code.strip("\n")
    code = re.sub(r"^```[\w#+-]*\n", "", code)
    return re.sub(r"\n?```\s*$", "", code)


async def regenerate_answer(tech: Technology, experience: Experience, question: Question, resume: str) -> Question:
    if not await answer_batch(tech, experience, [question], resume):
        raise AIError("The AI returned no answer")
    return question


# ---------- job runner ----------

class BankBuilder:
    def __init__(self, storage: Storage, index: QuestionIndex | None = None):
        self.storage = storage
        self.index = index
        self.jobs: dict[str, JobStatus] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def status(self, key: str) -> JobStatus:
        return self.jobs.get(key) or JobStatus(key=key)

    def is_running(self, key: str) -> bool:
        return self.status(key).state == "running"

    def start(self, tech: Technology, experience: Experience, mode: str) -> JobStatus:
        key = bank_key(tech.slug, experience.key)
        if self.is_running(key):
            return self.jobs[key]
        job = JobStatus(key=key, mode=mode, state="running", stage="Starting", started_at=now_iso())
        self.jobs[key] = job
        self._tasks[key] = asyncio.create_task(self._run(tech, experience, mode, job))
        return job

    async def _run(self, tech: Technology, experience: Experience, mode: str, job: JobStatus) -> None:
        key = job.key
        try:
            bank = None if mode == "build" else self.storage.load_bank(key)
            if bank is None:  # "build" = start over
                bank = Bank(key=key, slug=tech.slug, name=tech.name, kind=tech.kind, experience=experience.key)
                await self._forget_vectors(key)
            bank.name, bank.kind = tech.name, tech.kind
            bank.warnings = []  # notes describe the latest run only
            wanted = get_env_settings().questions_per_tech if mode == "build" else MORE_QUESTIONS
            if mode in ("build", "more") and ai.uses_claude_code():
                await self._generate(tech, experience, bank, wanted, job)  # no web search in this mode
            elif mode in ("build", "more"):
                await self._collect(tech, experience, bank, wanted, job, MAX_SEARCH_ROUNDS if mode == "build" else 2)
            await self._answer_missing(tech, experience, bank, job)
            missing = sum(1 for q in bank.questions if q.answer is None)
            job.state, job.stage = "done", "Done"
            job.message = f"{len(bank.questions)} questions" + (
                f", {missing} still without an answer" if missing else "")
        except AIError as exc:
            job.state, job.stage, job.message = "error", "Failed", str(exc)
        except Exception as exc:  # keep the server alive and show something useful
            log.exception("Build failed for %s", key)
            job.state, job.stage, job.message = "error", "Failed", f"Unexpected error: {exc}"
        finally:
            job.finished_at = now_iso()

    async def _forget_vectors(self, key: str) -> None:
        if self.index is not None and self.index.enabled:
            try:
                await self.index.delete_bank(key)
            except EmbeddingError:
                pass

    async def _collect(self, tech: Technology, experience: Experience, bank: Bank, wanted: int, job: JobStatus,
                       max_rounds: int) -> None:
        if not bank.categories:
            job.stage = "Planning categories"
            bank.categories = await plan_categories(tech, experience, wanted)
        start_count = len(bank.questions)
        target = start_count + wanted
        cap = start_count + math.ceil(wanted * MAX_OVERSHOOT)
        stats = {"pages": 0, "failed": 0, "not_on_page": 0, "off_level": 0, "repeats": 0}

        for _ in range(max_rounds):
            if len(bank.questions) >= target:
                break
            companies = self.storage.load_companies()
            queries = gather.search_queries(tech, experience, bank.search_rounds, companies)
            direct = gather.direct_pages(tech, bank.search_rounds, companies)  # known candidate-report pages
            bank.search_rounds += 1
            job.stage, job.done, job.total = "Searching the web for question pages", 0, 0
            found_refs, errors = await gather.discover_pages(queries)
            bank.warnings.extend(errors)
            used = set(bank.used_urls)
            refs = list({r.url: r for r in [*direct, *found_refs] if r.url not in used}.values())
            refs = refs[:MAX_PAGES_PER_ROUND]
            if not refs:
                continue

            job.stage = f"Reading {len(refs)} pages"
            pages, failures = await gather.fetch_pages(refs)
            pages = [p for p in pages if p.url not in used or p.url in {r.url for r in refs}]
            for url in [*(r.url for r in refs), *(p.url for p in pages)]:
                if url not in used:
                    used.add(url)
                    bank.used_urls.append(url)
            stats["pages"] += len(pages)
            stats["failed"] += len(failures)
            if failures:
                log.info("Could not read %d pages: %s", len(failures), failures)

            chunks = [doc for page in pages for doc in gather.split_page(page)]
            job.stage, job.done, job.total = "Picking out the questions on each page", 0, len(chunks)
            semaphore = asyncio.Semaphore(AI_CONCURRENCY)

            async def one(doc, semaphore=semaphore) -> list[Question]:
                async with semaphore:
                    try:
                        return await extract_from_chunk(tech, experience, bank.key, bank.categories, doc, stats)
                    except AIError as exc:
                        bank.warnings.append(f"Could not read questions from {doc.metadata['url']}: {exc}")
                        return []
                    finally:
                        job.done += 1

            found = [q for batch in await asyncio.gather(*(one(d) for d in chunks)) for q in batch]
            # Candidate-reported questions first, then ones that fit the experience level; basic ones last.
            found.sort(key=lambda q: (not q.reported, not q.fits_experience))
            job.stage = "Removing duplicates"
            new, merges = await semantic_dedupe(self.index, bank.key, bank.questions, found, bank.warnings)
            for duplicate, original in merges:  # same question on another site: more evidence it gets asked
                merge_into(original, duplicate)
            stats["repeats"] += len(merges)
            new = spread_by_page(new, MAX_PER_PAGE)[: max(0, cap - len(bank.questions))]
            bank.questions.extend(new)
            for q in bank.questions:
                q.likelihood = likelihood(q)
            if new and self.index is not None and self.index.enabled:
                try:
                    await self.index.add(bank.key, new)
                except EmbeddingError as exc:
                    bank.warnings.append(f"Questions were not added to the semantic index: {exc}")
            self.storage.save_bank(bank)
            job.message = f"{len(bank.questions)} questions so far"

        added = len(bank.questions) - start_count
        basic_kept = sum(1 for q in bank.questions[start_count:] if not q.fits_experience)
        bank.warnings.append(
            f"Search: read {stats['pages']} pages ({stats['failed']} could not be opened), added {added} "
            f"questions ({basic_kept} of them marked basic for {experience.label}); "
            f"{sum(1 for q in bank.questions if q.reported)} are reported by real candidates. "
            f"{stats['repeats']} repeats on other sites were merged in as extra evidence. Rejected "
            f"{stats['not_on_page']} the AI listed that were not on the page. {stats['off_level']} basic "
            "questions were found; they are used only after the ones that fit your level.")
        if len(bank.questions) < target:
            bank.warnings.append(f"Found {len(bank.questions)} real questions so far (target {target}). "
                                 "Use “Find more questions” to search more pages.")
        self.storage.save_bank(bank)
        if added == 0:
            raise AIError("No new questions were found on the pages searched — see the notes for details")

    async def _generate(self, tech: Technology, experience: Experience, bank: Bank, wanted: int,
                        job: JobStatus) -> None:
        """Claude Code mode: Claude Code writes the questions (marked AI-generated), a few categories per call."""
        if not bank.categories:
            job.stage = "Planning categories"
            bank.categories = await plan_categories(tech, experience, wanted)
        categories = [c for c in bank.categories if c != "Other"] or bank.categories
        start_count = len(bank.questions)
        per_category = max(3, math.ceil(wanted * 1.15 / len(categories)))
        groups = [categories[i:i + CATEGORIES_PER_CALL] for i in range(0, len(categories), CATEGORIES_PER_CALL)]
        companies = self.storage.load_companies()
        job.stage, job.done, job.total = "Claude Code is writing interview questions", 0, len(groups)
        semaphore = asyncio.Semaphore(AI_CONCURRENCY)
        errors: list[str] = []

        async def one(group: list[str]) -> list[Question]:
            async with semaphore:
                existing = {c: [q.question for q in bank.questions if q.category == c] for c in group}
                try:
                    result = await ai.generate_json(
                        prompt=prompts.generate_questions(tech, experience, group, per_category, existing, companies),
                        schema=GeneratedQuestions, temperature=0.5)
                except AIError as exc:
                    errors.append(str(exc))
                    return []
                finally:
                    job.done += 1
            created = now_iso()
            return [Question(id=question_id(bank.key, text), question=text,
                             category=match_category(g.category, bank.categories), level=normalize_level(g.level),
                             origin="ai", commonness=max(1, min(5, g.commonly_asked)), created_at=created)
                    for g in result.questions if (text := clean_question(g.question))]

        found = [q for batch in await asyncio.gather(*(one(g) for g in groups)) for q in batch]
        found.sort(key=lambda q: -q.commonness)
        job.stage = "Removing duplicates"
        new, _ = await semantic_dedupe(self.index, bank.key, bank.questions, found, bank.warnings)
        new = new[: math.ceil(wanted * MAX_OVERSHOOT)]
        bank.questions.extend(new)
        for q in bank.questions:
            q.likelihood = likelihood(q)
        if new and self.index is not None and self.index.enabled:
            try:
                await self.index.add(bank.key, new)
            except EmbeddingError as exc:
                bank.warnings.append(f"Questions were not added to the semantic index: {exc}")
        bank.warnings.append(f"Claude Code wrote {len(new)} questions (marked 🤖 AI-generated; not collected from "
                             "the web). Switch the provider in Settings to collect real interview questions.")
        if errors:
            bank.warnings.append(f"{len(errors)} of {len(groups)} question batches failed: {errors[0]}")
        self.storage.save_bank(bank)
        if len(bank.questions) == start_count:
            raise AIError(errors[0] if errors else "Claude Code returned no new questions")

    async def _answer_missing(self, tech: Technology, experience: Experience, bank: Bank, job: JobStatus) -> None:
        pending = [q for q in bank.questions if q.answer is None]
        if not pending:
            return
        resume = self.storage.load_resume()
        size = CLAUDE_CODE_ANSWER_BATCH if ai.uses_claude_code() else ANSWER_BATCH_SIZE  # fewer, bigger calls
        batches = [pending[i:i + size] for i in range(0, len(pending), size)]
        job.stage, job.done, job.total = f"Writing answers for {experience.label}", 0, len(pending)
        semaphore = asyncio.Semaphore(AI_CONCURRENCY)
        errors: list[str] = []

        async def one(batch: list[Question]) -> None:
            async with semaphore:
                try:
                    await answer_batch(tech, experience, batch, resume)
                except AIError as exc:
                    errors.append(str(exc))
                job.done += len(batch)
                self.storage.save_bank(bank)  # partial results show up in the UI right away

        await asyncio.gather(*(one(b) for b in batches))
        if errors:
            bank.warnings.append(f"{len(errors)} answer batch(es) failed: {errors[0]}")
            self.storage.save_bank(bank)
            if len(errors) == len(batches):
                raise AIError(f"Could not write answers: {errors[0]}")
