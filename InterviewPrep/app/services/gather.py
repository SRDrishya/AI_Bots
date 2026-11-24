"""Find real interview-question pages on the web and read them.

1. discover_pages  web-grounded AI search (Gemini + Google Search, or Claude web search) returns the
                   pages it found; we keep the page URLs from the search results, not the AI's text
2. fetch_page      downloads a page and turns its HTML into plain text (BeautifulSoup)
3. split_page      LangChain RecursiveCharacterTextSplitter cuts long pages into chunks for extraction
4. appears_in      checks an extracted question really is on the page, so the AI cannot invent one
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlparse

import httpx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import MAX_CHUNKS_PER_PAGE, PAGE_CONCURRENCY, Experience
from app.models import Technology
from app.services import ai, prompts
from app.services.ai import AIError

log = logging.getLogger(__name__)

FETCH_TIMEOUT = 20.0
MAX_PAGE_BYTES = 3_000_000
MIN_PAGE_CHARS = 600
CHUNK_SIZE = 12_000
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/130.0 Safari/537.36")

# Sites that never return question text to a plain download (video, login walls, social feeds).
SKIP_HOSTS = ("youtube.com", "youtu.be", "linkedin.com", "facebook.com", "instagram.com", "x.com",
              "twitter.com", "tiktok.com", "quora.com", "udemy.com", "coursera.org", "scribd.com")

# Search rounds, most useful first. Each round uses new queries so "find more" reaches new pages.
#  1. questions real candidates reported (AmbitionBox, Glassdoor, GeeksforGeeks interview experiences)
#  2. company rounds: reports from candidates at the target companies (3 companies per round)
#  3. well-known prep pages, used last
REPORT_QUERIES = [
    "{terms} interview questions asked ambitionbox {phrase}",
    "{terms} interview experience geeksforgeeks {phrase}",
    "{terms} interview questions glassdoor",
]
COMPANY_QUERY = "{company} {terms} interview questions asked {phrase}"
COMPANIES_PER_ROUND = 3
PREP_QUERY_SETS = [
    ["{terms} interview questions for {phrase}", "{terms} scenario based interview questions for experienced"],
    ["{terms} interview experience reddit {phrase}", "{terms} interview questions leetcode discuss"],
    ["top {terms} interview questions {year}", "{terms} tricky interview questions for experienced"],
    ["{terms} advanced interview questions {phrase} medium", "{terms} interview questions github list"],
]

_URL_IN_TEXT = re.compile(r"https?://[^\s<>()\[\]\"'|]+")
_WORDS = re.compile(r"[a-z0-9#+]+")


@dataclass
class PageRef:
    url: str
    title: str


@dataclass
class Page:
    url: str     # final URL after redirects
    title: str
    text: str


def search_rounds(companies: list[str]) -> list[list[str]]:
    """All query templates, one list per round, most valuable first."""
    company_rounds = [
        [COMPANY_QUERY.replace("{company}", c) for c in companies[k:k + COMPANIES_PER_ROUND]]
        for k in range(0, len(companies), COMPANIES_PER_ROUND)
    ]
    return [REPORT_QUERIES, *company_rounds, *PREP_QUERY_SETS]


AMBITIONBOX = "https://www.ambitionbox.com/interviews/"


def company_slug(company: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", company.lower()).strip("-")


def direct_pages(tech: Technology, round_no: int, companies: list[str]) -> list[PageRef]:
    """Known candidate-report pages read without searching, matched to the search rounds: round 0 the
    general role pages, then the same companies as that round's company searches. Missing pages (404)
    are simply skipped."""
    if not tech.ambitionbox_roles:
        return []
    if round_no == 0:
        return [PageRef(f"{AMBITIONBOX}{role}-interview-questions", f"AmbitionBox {role}")
                for role in tech.ambitionbox_roles]
    start = (round_no - 1) * COMPANIES_PER_ROUND
    if start >= len(companies):
        return []
    return [PageRef(f"{AMBITIONBOX}{company_slug(c)}-interview-questions/{role}", f"AmbitionBox {c} {role}")
            for c in companies[start:start + COMPANIES_PER_ROUND] for role in tech.ambitionbox_roles]


def search_queries(tech: Technology, experience: Experience, round_no: int, companies: list[str]) -> list[str]:
    terms = tech.search_terms or tech.name
    rounds = search_rounds(companies)
    template = rounds[round_no % len(rounds)]
    return [q.format(terms=terms, phrase=experience.phrase, year=date.today().year) for q in template]


def _usable(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        return False
    if url.lower().split("?")[0].endswith((".pdf", ".doc", ".docx", ".ppt", ".pptx", ".zip")):
        return False
    return not any(host == h or host.endswith("." + h) for h in SKIP_HOSTS)


async def discover_pages(queries: list[str]) -> tuple[list[PageRef], list[str]]:
    """Run each query through web-grounded AI search; return the pages found plus any error messages."""
    found: dict[str, PageRef] = {}
    errors: list[str] = []

    async def one(query: str) -> None:
        try:
            result = await ai.generate_grounded(prompt=prompts.discover(query))
        except AIError as exc:
            errors.append(f"Search '{query}' failed: {exc}")
            return
        for source in result.sources:
            if _usable(source.url):
                found.setdefault(source.url, PageRef(url=source.url, title=source.title))
        for url in _URL_IN_TEXT.findall(result.text):  # Claude/Gemini sometimes list extra pages in the text
            url = url.rstrip(".,;:")
            if _usable(url):
                found.setdefault(url, PageRef(url=url, title=urlparse(url).hostname or url))

    await asyncio.gather(*(one(q) for q in queries))
    return list(found.values()), errors


def html_to_text(html: str) -> tuple[str, str]:
    """-> (title, readable text). Keeps the article / main content when the page marks it."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = ""
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title = og["content"]
    elif soup.title and soup.title.string:
        title = soup.title.string
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "iframe"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    if len(root.get_text(strip=True)) < MIN_PAGE_CHARS:  # article tag too small: use the whole body
        root = soup.body or soup
    lines = (line.strip() for line in root.get_text("\n").splitlines())
    text = "\n".join(line for line in lines if line)
    return " ".join(title.split())[:200], text


def _fetchable_url(url: str) -> str:
    # new reddit renders with JavaScript; old.reddit.com serves the posts and comments as HTML
    return re.sub(r"^https?://(www\.)?reddit\.com/", "https://old.reddit.com/", url)


async def fetch_page(client: httpx.AsyncClient, ref: PageRef) -> Page | str:
    """The page as text, or a short reason it could not be read."""
    try:
        response = await client.get(_fetchable_url(ref.url))
    except httpx.HTTPError as exc:
        return f"{ref.url}: {type(exc).__name__}"
    if response.status_code >= 400:
        return f"{ref.url}: HTTP {response.status_code}"
    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "text/plain" not in content_type:
        return f"{ref.url}: not a web page ({content_type.split(';')[0] or 'unknown type'})"
    if len(response.content) > MAX_PAGE_BYTES:
        return f"{ref.url}: page too large"
    final_url = str(response.url)
    if "html" in content_type:
        title, text = html_to_text(response.text)
    else:
        title, text = "", response.text
    if len(text) < MIN_PAGE_CHARS:
        return f"{final_url}: no readable text (the site may need JavaScript or a login)"
    host = urlparse(final_url).hostname or final_url
    return Page(url=final_url, title=title or ref.title or host, text=text)


async def fetch_pages(refs: list[PageRef], transport: httpx.AsyncBaseTransport | None = None
                      ) -> tuple[list[Page], list[str]]:
    """Download pages in parallel. `transport` lets tests serve fake pages."""
    semaphore = asyncio.Semaphore(PAGE_CONCURRENCY)
    async with httpx.AsyncClient(follow_redirects=True, timeout=FETCH_TIMEOUT, transport=transport,
                                 headers={"User-Agent": USER_AGENT, "Accept-Language": "en"}) as client:
        async def one(ref: PageRef):
            async with semaphore:
                return await fetch_page(client, ref)

        results = await asyncio.gather(*(one(r) for r in refs))
    pages, failures, seen = [], [], set()
    for result in results:
        if isinstance(result, str):
            failures.append(result)
        elif result.url not in seen:  # two search links can redirect to the same page
            seen.add(result.url)
            pages.append(result)
    return pages, failures


_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=300)


def split_page(page: Page) -> list[Document]:
    docs = _splitter.split_documents([Document(page_content=page.text, metadata={"url": page.url,
                                                                                 "title": page.title})])
    return docs[:MAX_CHUNKS_PER_PAGE]


def _norm_words(text: str) -> list[str]:
    return _WORDS.findall(text.lower())


# Filler words ignored when matching questions (shared with duplicate detection).
STOPWORDS = frozenset(
    ["a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does", "did", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "can", "could", "would", "should", "will", "shall", "may", "might", "must", "in", "on", "at", "of", "for", "to", "from", "by", "with", "about", "into", "over", "and", "or", "but", "if", "then", "than", "this", "that", "these", "those", "it", "its", "you", "your", "we", "our", "they", "their", "i", "me", "my", "explain", "describe", "tell", "give", "example", "examples", "between", "difference", "differences", "use", "using", "used", "work", "works"])


def key_words(text: str) -> set[str]:
    return {w for w in _norm_words(text) if len(w) > 1 and w not in STOPWORDS}


def appears_in(question: str, text: str, min_coverage: float = 0.75) -> bool:
    """True if the question is really on the page: verbatim (ignoring case and punctuation), or at least
    75% of its key words appear close together in one passage. The looser check allows for removed
    numbering or an interview report ('they asked me about X') turned into a question, but rejects a
    question the AI made up from words scattered around the page."""
    q_words = _norm_words(question)
    if len(q_words) < 3:
        return False
    page_words = _norm_words(text)
    if " ".join(q_words) in " ".join(page_words):
        return True
    wanted = key_words(question)
    if not wanted:
        return False
    window = max(25, 3 * len(q_words))
    need = min_coverage * len(wanted)
    for start, word in enumerate(page_words):
        if word in wanted and len(wanted.intersection(page_words[start:start + window])) >= need:
            return True
    return False
