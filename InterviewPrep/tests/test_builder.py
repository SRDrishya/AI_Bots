import re
import zlib

import pytest

from app.config import get_experience
from app.models import (
    AnswerBatch,
    AnswerDraft,
    CategoryPlan,
    ExtractedQuestion,
    ExtractedQuestions,
    GeneratedQuestion,
    GeneratedQuestions,
    PlannedCategory,
    Question,
    SameQuestionVerdict,
    SameQuestionVerdicts,
)
from app.services import builder, gather
from app.services.catalog import DEFAULT_TECHNOLOGIES, find
from app.services.gemini import GroundedText, WebSource
from app.services.storage import Storage
from app.services.vectors import QuestionIndex
from tests.conftest import BagOfWordsEmbeddings

EXP = get_experience("5-7")


def q(text, category="Basics", key="t"):
    return Question(id=builder.question_id(key, text), question=text, category=category)


# ---------- helpers ----------

def test_clean_question_removes_numbering():
    assert builder.clean_question("Q12. what is LINQ?") == "What is LINQ?"
    assert builder.clean_question("  3) How does GC work? ") == "How does GC work?"


def test_dedupe_words_drops_rewordings_but_keeps_different_subjects():
    existing = [q("What is the difference between IEnumerable and IQueryable?")]
    new = [q("Explain the difference between IEnumerable and IQueryable."),
           q("What is the difference between IEnumerable and ICollection?"), q("Why?")]
    kept, merges = builder.dedupe_words(existing, new)
    assert [x.question for x in kept] == ["What is the difference between IEnumerable and ICollection?"]
    assert [(d.question, t.question) for d, t in merges] == [
        ("Explain the difference between IEnumerable and IQueryable.", existing[0].question)]


def test_match_category():
    cats = ["Async & Threading", "Memory", "Other"]
    assert builder.match_category("memory", cats) == "Memory"
    assert builder.match_category("Async", cats) == "Async & Threading"
    assert builder.match_category("Something else", cats) == "Other"


def test_filter_questions():
    a, b = q("Question about alpha things", "A"), q("Question about beta things", "B")
    b.level = "advanced"
    progress = {b.id: type("P", (), {"status": "review"})()}
    assert builder.filter_questions([a, b], progress, category="B") == [b]
    assert builder.filter_questions([a, b], progress, level="advanced") == [b]
    assert builder.filter_questions([a, b], progress, status="new") == [a]


# ---------- pipeline with a fake AI and fake web pages ----------

def page_text(n: int) -> str:
    lines = [f"Question {i}: How does feature{n}x{i} affect performance in production systems?" for i in range(12)]
    lines.append("A junior question: what does the keyword var mean?")
    return "Interview questions page\n" + "\n".join(lines) + "\nSome filler text. " * 30


class FakeAI:
    def __init__(self, grounded_fails=False, claude_code=False):
        self.grounded_fails = grounded_fails
        self.claude_code = claude_code
        self.prompts: list[str] = []
        self.searches = 0
        self.generated = 0

    def uses_claude_code(self):
        return self.claude_code

    async def generate_grounded(self, *, prompt, temperature=0.3):
        if self.claude_code:
            raise AssertionError("Claude Code mode must not use web search")
        if self.grounded_fails:
            raise builder.AIError("search not available")
        self.searches += 1
        n = self.searches
        return GroundedText(text="", sources=[WebSource(title=f"Page {n}-{i}", url=f"https://site{n}-{i}.example/q")
                                              for i in range(3)])

    async def generate_json(self, *, prompt, schema, temperature=0.3, fast=False):
        self.prompts.append(prompt)
        if schema is CategoryPlan:
            return CategoryPlan(categories=[PlannedCategory(name=f"Cat {i}", focus="f") for i in range(6)])
        if schema is ExtractedQuestions:
            text = prompt.split("<page_text>", 1)[1]
            reported = "interview report" in text
            found = [ExtractedQuestion(question=m, category="Cat 1", level="advanced", fits_experience=True,
                                       reported_by_candidate=reported, companies=["Infosys"] if reported else [])
                     for m in re.findall(r"How does feature\S+ affect performance in production systems\?", text)]
            found.append(ExtractedQuestion(question="What does the keyword var mean?", category="Cat 0",
                                           level="beginner", fits_experience=False, reported_by_candidate=False,
                                           companies=[]))
            found.append(ExtractedQuestion(question="Invented: how do quantum blockchains scale Kubernetes?",
                                           category="Cat 2", level="advanced", fits_experience=True,
                                           reported_by_candidate=False, companies=[]))
            return ExtractedQuestions(questions=found)
        if schema is GeneratedQuestions:
            cats = re.search(r"for EACH of these categories: (.+?)\.\n", prompt).group(1).split(", ")
            per_cat = int(re.search(r"Write (\d+) distinct questions", prompt).group(1))
            items = []
            for cat in cats:
                for _ in range(per_cat):
                    self.generated += 1
                    n = self.generated
                    items.append(GeneratedQuestion(question=f"How would you handle scenario{n} involving topic{n}?",
                                                   category=cat, level="advanced", commonly_asked=1 + n % 5))
            return GeneratedQuestions(questions=items)
        if schema is AnswerBatch:
            count = prompt.split("Questions:\n", 1)[1].count("\n") + 1
            return AnswerBatch(answers=[
                AnswerDraft(number=n, summary="Short.", explanation="Long **answer**.", code_language="csharp",
                            code_example="```csharp\nvar x = 1;\n```", key_points=["k"], pitfalls=["p"],
                            follow_ups=["f"], your_angle="")
                for n in range(1, count + 1)])
        if schema is SameQuestionVerdicts:
            count = prompt.count("\n   B: ")
            return SameQuestionVerdicts(verdicts=[SameQuestionVerdict(pair=i + 1, same=False) for i in range(count)])
        raise AssertionError(schema)


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setenv("QUESTIONS_PER_TECH", "20")
    store = Storage(tmp_path)
    store.save_resume("Lead engineer. C#, .NET")
    return store


@pytest.fixture
def fake_web(monkeypatch):
    fake = FakeAI()
    monkeypatch.setattr(builder, "ai", fake)
    monkeypatch.setattr(gather, "ai", fake)

    async def fake_fetch(refs, transport=None):
        pages = [gather.Page(url=r.url, title=r.title, text=page_text(zlib.crc32(r.url.encode()))) for r in refs]
        return pages, ["https://blocked.example: HTTP 403"]

    monkeypatch.setattr(gather, "fetch_pages", fake_fetch)
    return fake


async def run_job(b, tech, mode, experience=EXP):
    job = b.start(tech, experience, mode)
    await b._tasks[job.key]
    return job


async def test_build_keeps_only_real_questions_that_fit_the_experience(storage, fake_web):
    tech = find(DEFAULT_TECHNOLOGIES, "csharp")
    job = await run_job(builder.BankBuilder(storage), tech, "build")
    assert job.state == "done", job.message
    bank = storage.load_bank("csharp--5-7")
    assert 20 <= len(bank.questions) <= 24  # target 20, at most 20% over
    texts = [x.question for x in bank.questions]
    assert not any("Invented" in t for t in texts)          # not on the page -> rejected
    assert not any("keyword var" in t for t in texts)       # basic for 5-7 years -> only fills spare room
    assert all(x.fits_experience for x in bank.questions)
    page_urls = {x.sources[0].url for x in bank.questions}
    assert any(u.startswith("https://www.ambitionbox.com/interviews/") for u in page_urls)  # direct report pages
    assert any(u.startswith("https://site") for u in page_urls)                            # search results
    pages = [x.sources[0].url for x in bank.questions]
    assert max(pages.count(u) for u in page_urls) <= builder.MAX_PER_PAGE                 # no page dominates
    assert all(x.answer and x.answer.code_example == "var x = 1;" for x in bank.questions)
    assert any("not on the page" in w for w in bank.warnings)
    answer_prompts = [p for p in fake_web.prompts if "Write model interview answers" in p]
    assert answer_prompts and all("5–7 years" in p for p in answer_prompts)


async def test_no_ai_written_questions_when_web_search_fails(storage, monkeypatch):
    fake = FakeAI(grounded_fails=True)
    monkeypatch.setattr(builder, "ai", fake)
    monkeypatch.setattr(gather, "ai", fake)
    job = await run_job(builder.BankBuilder(storage), find(DEFAULT_TECHNOLOGIES, "react"), "build")
    assert job.state == "error"
    bank = storage.load_bank("react--5-7")
    assert bank.questions == []
    assert any("search not available" in w for w in bank.warnings)


async def test_more_mode_reads_new_pages_only(storage, fake_web):
    b = builder.BankBuilder(storage)
    tech = find(DEFAULT_TECHNOLOGIES, "python")
    await run_job(b, tech, "build")
    before = storage.load_bank("python--5-7")
    job = await run_job(b, tech, "more")
    assert job.state == "done", job.message
    after = storage.load_bank("python--5-7")
    assert len(after.questions) > len(before.questions)
    assert after.search_rounds > before.search_rounds
    assert len(set(after.used_urls)) == len(after.used_urls)


async def test_experience_levels_have_separate_banks(storage, fake_web):
    b = builder.BankBuilder(storage)
    tech = find(DEFAULT_TECHNOLOGIES, "linq")
    await run_job(b, tech, "build")
    await run_job(b, tech, "build", get_experience("0-2"))
    assert storage.load_bank("linq--5-7") and storage.load_bank("linq--0-2")
    assert storage.load_bank("linq--0-2").questions[0].id.startswith("linq--0-2-")


# ---------- semantic duplicates (LangChain + Chroma with offline embeddings) ----------

async def test_semantic_dedupe_search_related_and_delete(tmp_path, monkeypatch):
    index = QuestionIndex(tmp_path / "chroma", embeddings=BagOfWordsEmbeddings())
    key = "csharp--5-7"
    existing = [q("What is dependency injection in .NET Core?", key=key),
                q("How do you find the second highest salary in SQL?", key=key)]
    # existing questions are NOT pre-indexed: semantic_dedupe must index them itself before comparing

    judged = []

    async def fake_judge(pairs):
        judged.extend(c.question for c, _ in pairs)
        return {c.id for c, _ in pairs if "pay" in c.question}

    monkeypatch.setattr(builder, "judge_duplicates", fake_judge)
    candidates = [q("Explain DI in dotnet core", key=key),                    # same meaning -> auto duplicate
                  q("Query to get the 2nd top pay in SQL", key=key),          # close -> AI confirms duplicate
                  q("What are records in C# and when to use them?", key=key)]  # new
    assert len(builder.dedupe_words(existing, candidates)[0]) == 3  # word matching alone misses the rewordings
    kept, merges = await builder.semantic_dedupe(index, key, existing, candidates, [])
    assert {(d.question, t.question) for d, t in merges} == {
        ("Explain DI in dotnet core", existing[0].question),
        ("Query to get the 2nd top pay in SQL", existing[1].question)}
    assert [x.question for x in kept] == ["What are records in C# and when to use them?"]
    assert judged == ["Query to get the 2nd top pay in SQL"]  # only the borderline pair went to the AI

    await index.add(key, kept)
    hits = await index.search(key, "records C#", k=3)
    assert hits[0][0] == kept[0].id
    related = await index.related(key, existing[0].id, k=2)
    assert existing[0].id not in [i for i, _ in related]
    await index.delete_bank(key)
    assert await index.search(key, "records", k=3) == []


# ---------- "how likely is it to be asked" ----------

def test_merge_into_collects_sites_and_companies():
    target = q("What is middleware in ASP.NET Core?")
    target.sources = [builder.Source(title="Blog", url="https://blog.example/a")]
    dup = q("Explain middleware in ASP.NET Core.")
    dup.sources = [builder.Source(title="AmbitionBox", url="https://www.ambitionbox.com/x")]
    dup.reported, dup.companies = True, ["Infosys"]
    builder.merge_into(target, dup)
    builder.merge_into(target, dup)  # merging the same page twice adds nothing
    assert [s.url for s in target.sources] == ["https://blog.example/a", "https://www.ambitionbox.com/x"]
    assert target.reported and target.companies == ["Infosys"]


def test_likelihood_ranks_real_interview_questions_first():
    blog = q("Blog-only question about something")
    blog.sources = [builder.Source(title="b", url="https://blog.example/1")]
    reported = q("Question a candidate reported at Infosys")
    reported.sources = [builder.Source(title="a", url="https://www.ambitionbox.com/1"),
                        builder.Source(title="g", url="https://www.glassdoor.co.in/1")]
    reported.reported, reported.companies = True, ["Infosys", "TCS"]
    basic = q("Basic question")
    basic.fits_experience = False
    assert builder.likelihood(reported) > builder.likelihood(blog) > builder.likelihood(basic)


async def test_build_puts_reported_questions_first_and_merges_repeats(storage, fake_web, monkeypatch):
    real_fetch = gather.fetch_pages  # the fake_web version

    async def fetch_with_report(refs, transport=None):
        pages, failures = await real_fetch(refs)
        # the first page is a candidate's interview report, and another site copied it word for word
        pages[0].text = "Candidate interview report\n" + pages[0].text
        pages.append(gather.Page(url="https://mirror.example/copy", title="Mirror", text=pages[0].text))
        return pages, failures

    monkeypatch.setattr(gather, "fetch_pages", fetch_with_report)
    job = await run_job(builder.BankBuilder(storage), find(DEFAULT_TECHNOLOGIES, "lead-dotnet-fullstack"), "build")
    assert job.state == "done", job.message
    bank = storage.load_bank("lead-dotnet-fullstack--5-7")
    top = max(bank.questions, key=lambda x: x.likelihood)
    assert top.reported and top.companies == ["Infosys"]
    assert len({s.url for s in top.sources}) == 2          # same question on the mirror page merged in
    assert any("merged in as extra evidence" in w for w in bank.warnings)


def test_spread_by_page_gives_every_page_a_turn():
    def on(page, n):
        x = q(f"Question {n} from {page} about topic{n}")
        x.sources = [builder.Source(title=page, url=page)]
        return x
    qs = [on("a", i) for i in range(5)] + [on("b", i) for i in range(2)]
    order = [x.sources[0].url for x in builder.spread_by_page(qs, 2)]
    assert order == ["a", "a", "b", "b", "a", "a", "a"]


def test_report_sites_rank_above_articles():
    article, report = q("Same kind of question one"), q("Same kind of question two")
    article.sources = [builder.Source(title="blog", url="https://blog.example/x")]
    report.sources = [builder.Source(title="ab", url="https://www.ambitionbox.com/interviews/x")]
    article.reported = report.reported = True
    assert builder.likelihood(report) > builder.likelihood(article)


# ---------- Claude Code mode: AI-written questions ----------

async def test_claude_code_mode_writes_labelled_questions_without_web_search(storage, monkeypatch):
    fake = FakeAI(claude_code=True)
    monkeypatch.setattr(builder, "ai", fake)
    monkeypatch.setattr(gather, "ai", fake)
    job = await run_job(builder.BankBuilder(storage), find(DEFAULT_TECHNOLOGIES, "csharp"), "build")
    assert job.state == "done", job.message
    bank = storage.load_bank("csharp--5-7")
    assert 20 <= len(bank.questions) <= 24
    assert all(x.origin == "ai" and not x.sources and x.answer for x in bank.questions)
    assert fake.searches == 0
    ranked = sorted(bank.questions, key=lambda x: -x.likelihood)
    assert ranked[0].commonness >= ranked[-1].commonness      # most commonly asked first
    assert any("AI-generated" in w for w in bank.warnings)
    answer_calls = [p for p in fake.prompts if "Write model interview answers" in p]
    assert len(answer_calls) == -(-len(bank.questions) // builder.CLAUDE_CODE_ANSWER_BATCH)  # bigger batches
    question_calls = [p for p in fake.prompts if "MOST COMMONLY asked" in p]
    assert len(question_calls) == 2                            # 6 categories, 3 per call


async def test_claude_code_more_avoids_repeating_existing_questions(storage, monkeypatch):
    fake = FakeAI(claude_code=True)
    monkeypatch.setattr(builder, "ai", fake)
    b = builder.BankBuilder(storage)
    tech = find(DEFAULT_TECHNOLOGIES, "sql-server")
    await run_job(b, tech, "build")
    first = storage.load_bank("sql-server--5-7").questions[0].question
    await run_job(b, tech, "more")
    more_prompts = [p for p in fake.prompts if "MOST COMMONLY asked" in p][-2:]
    assert any(first in p for p in more_prompts)               # existing questions listed as "do not repeat"
