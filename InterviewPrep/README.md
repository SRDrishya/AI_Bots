# InterviewPrep

A personal, local web app to help you **clear your interviews**. It collects the questions real candidates say they were asked, for each technology on your resume and for your target roles. It ranks them by how likely they are to come up, and writes answers pitched at your experience level.

- **Questions come only from real pages.** The AI never writes the questions, and every question links to the page it came from.
- **Real interview reports come first.** AmbitionBox company pages ("asked in TCS"), Glassdoor interviews and GeeksforGeeks interview experiences are read before prep articles.
- **Ranked by how likely they are to be asked**, with a **🎯 Top 30 must-know** view for last-minute revision.
- **Answers are written for your experience level**: 0–2, 3–4, **5–7 (default)**, 8–10 or 11+ years.
- **Review everything on the page first**, then select questions and generate a **PDF, Word (DOCX) or HTML** file.

## Question banks

| Group | Banks |
|---|---|
| **Target roles** | **Lead .NET Full Stack Developer**, **.NET Full Stack + Gen AI Developer**: whole-interview banks covering every skill, plus **Coding Problems** (DSA) and **HR & Behavioral** |
| Languages | C#, Python, JavaScript, HTML & CSS |
| Backend | .NET Core / .NET 8-9, ASP.NET MVC, ASP.NET Core Web API, FastAPI |
| Frontend | ReactJS |
| Data | Entity Framework (EF Core & EF6), LINQ, ADO.NET, SQL Server, PostgreSQL |
| Tools & DevOps | Azure DevOps & CI/CD, Git & GitHub, Unit Testing (xUnit & Moq) |
| Interview rounds | System Design & Microservices, Design Patterns & SOLID, Behavioral & Leadership, AI / LLM & RAG |

Use **+ Add** for anything else (e.g. Docker, Kafka).

## Where the questions come from

Each run goes through search rounds, most valuable first:

1. **Candidate interview reports.** AmbitionBox role pages are read directly (e.g. `ambitionbox.com/interviews/dot-net-developer-interview-questions`), and the app searches for AmbitionBox, Glassdoor and GeeksforGeeks interview experiences.
2. **Your target companies.** For each company, the app reads its AmbitionBox page for your role (e.g. `…/infosys-interview-questions/dot-net-developer`) and searches for its interview reports, 3 companies per round. The default list is IT services (TCS, Infosys, Cognizant, Accenture, Wipro, HCLTech, Capgemini, LTIMindtree, Virtusa), product companies (Zoho, Freshworks, Chargebee, PayPal, Microsoft, Amazon) and banking / insurance tech (Citi, Barclays, Wells Fargo, Standard Chartered, HSBC). Edit it in **⚙ Settings**.
3. **Prep sites.** Well-known question pages, used last.

For each page:
- The AI copies out the questions on it. Each one is **checked against the page text, and rejected if it isn't there**.
- Vague entries ("Basic Dotnet question") are skipped.
- The app records whether the page reports it as asked in a real interview, and at which companies.
- No page contributes more than 6 new questions per round, so no single site dominates.

**Duplicates become evidence.** When the same question shows up on another site, it isn't stored twice: the new source and company are added to the existing question. Duplicates are found by matching words, then by meaning (LangChain + Chroma with Gemini embeddings); close calls go to a quick AI check.

**Likelihood score** (used for sorting and Top 30), adding up:
- reported by a real candidate: +3
- found on a candidate-report site: +1
- each named company: +1, up to 3
- each extra site with the same question: +1.5, up to 4 sites
- typical for your experience level: +1

On each question, chips show 🎤 *real interview: TCS, Infosys* and 🔥 *found on N sites*.

**Answers** are written for your experience level:
- technical questions: how it works, trade-offs, a code example, pitfalls and follow-up questions;
- coding problems: the approach, its complexity, and a C# solution;
- behavioral questions: STAR format from your resume;
- HR questions: a framing you can adapt.

Each answer has a **Your angle** note linking it to your resume. It never claims experience your resume doesn't show.

## AI provider (⚙ Settings)

| Provider | Questions | Answers | Needs |
|---|---|---|---|
| **Auto** (default) | Collected from real interview reports on the web | Gemini, falling back to the Claude API | Gemini and/or Anthropic API key |
| Gemini only / Claude API only | Collected from the web | That provider | Its API key |
| **Claude Code** | **Written by Claude Code**, marked **🤖 AI-generated** | Claude Code | Claude Code signed in to your Claude subscription. No API credits needed. |

Choose **Claude Code** when API tokens run out. The app runs `claude -p` in headless mode and keeps each call lean:
- no tools, MCP servers, hooks or saved sessions;
- a short system prompt instead of Claude Code's large default one, which costs about 20× less usage per call;
- questions and answers are batched (3 categories, or 8 answers, per call).

It also removes `ANTHROPIC_API_KEY` from Claude Code's environment, so your sign-in is used instead of the API account.

Questions are ranked by how commonly the AI says they come up (1–5). The Notes box and every export say they're AI-generated. Claude Code is found automatically, either on PATH or bundled with the VS Code extension. Set `CLAUDE_CODE_PATH` or `CLAUDE_CODE_MODEL` in `.env` to override. In a live test, 15 Entity Framework questions with answers took about 3.5 minutes.

## Semantic features (LangChain + Chroma)

- **Search by meaning**: press Enter in the search box or click 🧠. For example, "why are my database calls slow" finds questions about connection pooling, async and batch updates.
- **Related questions**: shown under each answer.
- **Duplicate removal**: see step 4 above.

These need a Gemini key. Without one, or when Gemini is rate-limited, the app falls back to word matching and says so in the Notes.

## Setup

Requires Python 3.11+ (developed on 3.14) and at least one AI key. They're the same keys JobSearchAgent uses.

```powershell
cd InterviewPrep
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env      # then fill in GEMINI_API_KEY and/or ANTHROPIC_API_KEY
.venv\Scripts\python -m app
```

Open <http://127.0.0.1:8010>. Your resume text is imported from `../JobSearchAgent/data/resume_text.json` on first start. You can change it in **⚙ Settings**.

## Using it

1. Pick a **Role** (a whole interview) or a **Technology**, and an **Experience** level, at the top.
   Choose how many questions to add per run with the **Questions** selector (25 / 50 / 100 / 150 / 200);
   it applies to the first run and to **Find more questions**.
2. Click **Collect questions from the web**. A full run takes several minutes, and questions appear as they're found.
3. In **Browse & review**, read every question and answer. Filter by category, level or status, and search by text or by meaning. Mark each one *I know this* or *Review later*, regenerate weak answers, and delete ones you don't want.
4. Tick the questions you want, or use **Select all shown**. Then click **Review & export…**, check the list, pick **PDF / DOCX / HTML**, and click **Confirm & generate**.

**Practice tab**: you get one question at a time. Type your answer to get a 0–10 score judged against your experience level, what was good, what was missing, and an improved version of your answer.

## Settings (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` | — | At least one is required. Gemini is also needed for semantic search. |
| `AI_PROVIDER` | `gemini` | Provider tried first; the other one is the fallback |
| `GEMINI_MODEL`, `GEMINI_FALLBACK_MODELS` | `gemini-3.8-flash`, `gemini-3.5-flash-lite` | |
| `ANTHROPIC_MODEL`, `ANTHROPIC_FAST_MODEL` | `claude-opus-5-5`, `claude-haiku-5-5` | The fast model handles extraction, duplicate checks and scoring |
| `EMBEDDING_MODEL` | `models/gemini-embedding-2` | Embeddings for the Chroma index |
| `QUESTIONS_PER_TECH` | `100` | Target size of a full run |
| `PORT` | `8010` | |

**Quotas (important):** on the free tier, `gemini-3.8-flash` allows only **20 requests per day**, and a full run needs 100+. The app works around this:
- when a model's daily quota is used up, it switches to `gemini-3.5-flash-lite` for 6 hours;
- when a per-minute limit is hit, it waits and retries;
- if both fail, it falls back to Claude, which only works if your Anthropic account has credit.

For best answer quality, enable billing on the Gemini key or add Anthropic credit. Questions that fail to get an answer can be retried later with **Fill missing answers**.

## Data

Everything is stored locally in `data/` (gitignored):

- `banks/<technology>--<experience>.json`: questions, answers, source links, and pages already read
- `chroma/`: the vector index
- `progress.json`: Known/Review status and practice attempts
- `technologies.json`: technologies you added
- `resume.txt`: your resume text

## Tests

```powershell
.venv\Scripts\python -m pytest
```

The tests use a fake AI, fake web pages and offline embeddings, so they make no network calls.
