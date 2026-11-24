import httpx
import pytest

from app.config import get_experience
from app.services import gather
from app.services.catalog import DEFAULT_TECHNOLOGIES, find
from app.services.gemini import GroundedText, WebSource

PAGE = """
<html><head><title>Top 50 C# Interview Questions</title><script>var x = 'What is a hidden question?';</script></head>
<body><nav>Home | About | Login</nav>
<article>
  <h1>C# interview questions for 5 years experience</h1>
  <p>Q12. What is the difference between IEnumerable and IQueryable?</p>
  <p>Answer: IQueryable builds an expression tree ...</p>
  <p>13) How does async/await work under the hood in C#?</p>
  <p>In my interview at a product company they asked me about the garbage collector generations and when Gen 2 runs.</p>
  <p>Lots more filler text about the site and courses. """ + "Filler sentence about learning. " * 40 + """</p>
</article>
<footer>Copyright</footer></body></html>
"""


def test_html_to_text_keeps_article_and_drops_scripts_and_nav():
    title, text = gather.html_to_text(PAGE)
    assert title == "Top 50 C# Interview Questions"
    assert "IEnumerable and IQueryable" in text
    assert "hidden question" not in text and "Login" not in text and "Copyright" not in text


def test_appears_in_accepts_page_questions_and_rejects_invented_ones():
    _, text = gather.html_to_text(PAGE)
    assert gather.appears_in("What is the difference between IEnumerable and IQueryable?", text)
    assert gather.appears_in("How does async/await work under the hood in C#?", text)
    # an interview report turned into a question with the page's words
    assert gather.appears_in("When does the garbage collector run Gen 2 generations?", text)
    # made-up question built from words scattered over the page
    assert not gather.appears_in("How do you learn IQueryable courses on the site with async generations?", text)
    assert not gather.appears_in("What is dependency injection in ASP.NET Core?", text)


def test_search_rounds_start_with_interview_reports_then_companies():
    tech, exp = find(DEFAULT_TECHNOLOGIES, "csharp"), get_experience("5-7")
    companies = ["Infosys", "TCS", "Citi", "Zoho"]
    first = gather.search_queries(tech, exp, 0, companies)
    assert any("ambitionbox" in q and "5 years experience" in q for q in first)
    assert gather.search_queries(tech, exp, 1, companies)[0].startswith("Infosys C# interview questions")
    assert gather.search_queries(tech, exp, 2, companies) == ["Zoho C# interview questions asked 5 years experience"]
    later = gather.search_queries(tech, exp, 3, companies)
    assert all(c not in " ".join(later) for c in companies)  # then general prep pages
    assert len(gather.search_rounds(companies)) == 1 + 2 + len(gather.PREP_QUERY_SETS)


async def test_discover_pages_keeps_search_sources_and_skips_video_sites(monkeypatch):
    async def fake_grounded(*, prompt, temperature=0.3):
        return GroundedText(text="- Extra | https://example.org/more-questions.\n- Video | https://youtube.com/watch?v=1",
                            sources=[WebSource(title="Site A", url="https://a.example.com/q"),
                                     WebSource(title="Doc", url="https://b.example.com/file.pdf")])

    monkeypatch.setattr(gather.ai, "generate_grounded", fake_grounded)
    refs, errors = await gather.discover_pages(["q1", "q2"])
    assert errors == []
    assert sorted(r.url for r in refs) == ["https://a.example.com/q", "https://example.org/more-questions"]


async def test_fetch_pages_reports_failures_and_follows_redirects():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "redirect.example":
            return httpx.Response(302, headers={"location": "https://good.example/page"})
        if request.url.host == "good.example":
            return httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})
        if request.url.host == "old.reddit.com":
            return httpx.Response(200, text=PAGE, headers={"content-type": "text/html; charset=utf-8"})
        return httpx.Response(403)

    refs = [gather.PageRef("https://redirect.example/x", "r"), gather.PageRef("https://good.example/page", "g"),
            gather.PageRef("https://blocked.example/", "b"), gather.PageRef("https://www.reddit.com/r/csharp/1", "rd")]
    pages, failures = await gather.fetch_pages(refs, transport=httpx.MockTransport(handler))
    assert sorted(p.url for p in pages) == ["https://good.example/page", "https://old.reddit.com/r/csharp/1"]
    assert failures == ["https://blocked.example/: HTTP 403"]


def test_split_page_limits_chunks():
    page = gather.Page(url="u", title="t", text="Question line here.\n" * 20000)
    docs = gather.split_page(page)
    assert len(docs) == gather.MAX_CHUNKS_PER_PAGE
    assert docs[0].metadata == {"url": "u", "title": "t"}


def test_direct_ambitionbox_pages_follow_the_search_rounds():
    tech = find(DEFAULT_TECHNOLOGIES, "lead-dotnet-fullstack")
    companies = ["Infosys", "Standard Chartered", "TCS", "Zoho"]
    first = [r.url for r in gather.direct_pages(tech, 0, companies)]
    assert "https://www.ambitionbox.com/interviews/dot-net-developer-interview-questions" in first
    second = [r.url for r in gather.direct_pages(tech, 1, companies)]
    assert "https://www.ambitionbox.com/interviews/standard-chartered-interview-questions/dot-net-developer" in second
    assert not any("zoho" in u for u in second)  # Zoho is in the next company round
    assert gather.direct_pages(tech, 9, companies) == []


async def test_gemini_waits_out_rate_limits_then_succeeds(monkeypatch):
    from app.services import gemini

    waits, calls = [], []

    async def fake_sleep(seconds):
        waits.append(seconds)

    async def flaky(model):
        calls.append(model)
        if len(calls) < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
        return "ok"

    monkeypatch.setattr(gemini.asyncio, "sleep", fake_sleep)
    assert await gemini._run_chain("m", flaky) == "ok"
    assert waits == list(gemini.RATE_LIMIT_WAITS) and len(calls) == 3


async def test_gemini_bad_key_fails_immediately(monkeypatch):
    from app.services import gemini

    async def bad(model):
        raise RuntimeError("API key not valid")

    with pytest.raises(gemini.GeminiError, match="key rejected"):
        await gemini._run_chain("m", bad)


async def test_gemini_daily_quota_moves_to_the_fallback_model(monkeypatch):
    from app.services import gemini

    gemini.reset_cooldowns()
    monkeypatch.setattr(gemini, "model_chain", lambda m: [m, "lite"])
    calls = []

    async def per_model(model):
        calls.append(model)
        if model == "main":
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'")
        return f"answered by {model}"

    assert await gemini._run_chain("main", per_model) == "answered by lite"
    assert calls == ["main", "lite"]  # no waiting on a daily limit
    gemini.reset_cooldowns()
