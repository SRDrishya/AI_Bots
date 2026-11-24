# JobPilot AI

A personal job-search assistant that runs on your own computer. Upload your resume once and it:

- searches several job sources for **Chennai jobs and remote jobs open to India**
- removes jobs you can't take (e.g. "Remote — USA only") and merges duplicates
- scores every job against your resume (with Gemini AI re-scoring the best matches)
- runs a **daily search at 08:00 IST** and marks jobs you haven't seen before as **NEW**
- tracks your applications (saved → applied → interviewing → offer / rejected)
- exports results to **Excel / CSV** and writes **tailored cover letters**
- rewrites your saved resume for a specific job (**📄 Tailor resume**): editable preview, then a one-page **PDF / DOCX**

Everything (profile, results, tracker) is stored locally in the `data/` folder.

---

## 1. Requirements

- **Python 3.11 or newer** (developed on Python 3.14) — check with `python --version`
- Internet access
- Optional API keys (the app works without them, with fewer results — see step 3)

## 2. Install (first time only)

Open a terminal in the project folder (`JobSearchAgent`).

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 3. Configure API keys

Copy the example settings file to `.env`:

```powershell
copy .env.example .env        # Windows
```
```bash
cp .env.example .env          # macOS / Linux
```

Then open `.env` and fill in the keys you have. **All are optional:**

| Setting | What it adds | Where to get it | Free tier |
|---|---|---|---|
| `GEMINI_API_KEY` | AI resume parsing, AI match scoring, cover letters, tailored resumes | https://aistudio.google.com/apikey | yes |
| `ANTHROPIC_API_KEY` | Same AI features using Claude, instead of or as a fallback for Gemini (company research uses Claude's live web search) | https://console.anthropic.com → API keys | no (paid per use) |
| `RAPIDAPI_KEY` | **JSearch** — Chennai jobs from LinkedIn, Naukri, Shine, company sites | https://rapidapi.com → subscribe to *JSearch* | 200 calls/month |
| `SERPAPI_KEY` | **Google Jobs** — Chennai jobs | https://serpapi.com | 250 searches/month |
| `ADZUNA_APP_ID` + `ADZUNA_APP_KEY` | **Adzuna India** | https://developer.adzuna.com | yes |
| `GOOGLE_OAUTH_CLIENT_ID` + `GOOGLE_OAUTH_CLIENT_SECRET` | Connect a Gmail inbox for job-alert emails | Google Cloud Console (see below) | yes |
| `MICROSOFT_OAUTH_CLIENT_ID` + `MICROSOFT_OAUTH_CLIENT_SECRET` | Connect a Hotmail/Outlook inbox for job-alert emails | Microsoft Entra admin center (see below) | yes |

Without any keys you still get remote jobs from Himalayas, Remotive, Jobicy, Arbeitnow and company
career boards, scored locally — but very few Chennai-office jobs. **JSearch and SerpAPI are the
sources that find Chennai jobs**, so they are strongly recommended.

> **Keep `.env` private.** It holds your keys and is excluded from git. Never put real keys in
> `.env.example` — that file is meant to be shared (a test fails if it contains a key).

Other settings in `.env` (defaults are fine):

| Setting | Default | Meaning |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.8-flash` | Main Gemini model |
| `GEMINI_FALLBACK_MODELS` | `gemini-3.5-flash-lite` | Used automatically if the main model is overloaded |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | Claude model for writing: cover letters, tailored resumes, company research |
| `ANTHROPIC_FAST_MODEL` | `claude-haiku-5-5` | Cheaper Claude model for resume parsing and job match scoring |
| `AI_PROVIDER` | `gemini` | `gemini` or `anthropic`: which AI to try first. If it fails (quota, rate limit, overload) and the other has a key, the app switches automatically |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Where the web app is served |
| `DATA_DIR` | `./data` | Where your profile, results and tracker are saved |

### Email jobs: Gmail and Hotmail/Outlook

The **Email jobs** tab searches connected inboxes for job-related messages from LinkedIn, Indeed
and Naukri, then lists the message, date, preview and detected job links. It uses read-only OAuth;
it never asks for your mailbox password, changes messages, or sends email contents to an AI service.
OAuth refresh tokens are kept in the operating system's credential store; provider client secrets
belong only in your private `.env`.

Set up OAuth once for each provider:

1. **Google:** In Google Cloud Console, create/select a project, enable the Gmail API, configure
   the OAuth consent screen (add your Gmail account as a test user if the app is in testing), and
   create an OAuth client ID of type **Web application**. Add this authorized redirect URI:
   `http://127.0.0.1:8000/api/emails/oauth/google/callback`.
2. **Microsoft:** In Microsoft Entra admin center, register an app that supports personal Microsoft
   accounts. Add a **Web** redirect URI:
   `http://127.0.0.1:8000/api/emails/oauth/microsoft/callback`. Create a client secret and add
   Microsoft Graph's delegated `User.Read` and `Mail.Read` permissions.
3. Copy each provider's client ID and secret into the matching `GOOGLE_OAUTH_*` and
   `MICROSOFT_OAUTH_*` entries in `.env`, then restart the app. In the **Email jobs** tab, connect
   Gmail and Hotmail/Outlook separately, approve read-only access, choose a date range and click
   **Search emails**.

Email checks are incremental: each account remembers when it was last checked, and the next check
only reads mail received after that (with a 1-hour overlap so nothing slips through). The first
check of a newly connected account looks back 7 days. The daily scheduled search also checks for new
job emails, so normally you just open the tab and see the day's new ones marked **New**. Found
emails are saved locally (`data/email_scan.json`) for 60 days; **Rescan** re-reads the last 7/30/90
days if you ever need to. Disconnecting an account removes its saved results.

Both inboxes are searched in parallel. If one fails (for example an expired sign-in), the other's
results are still listed with an error note for the failed account — reconnect it to fix. LinkedIn
social mail (connection invites, newsletters, group posts) is skipped; Indeed employer replies
(`indeedemail.com`) are included.

If you run the app on another port, the redirect base defaults to `http://127.0.0.1:<PORT>`. Set
`EMAIL_OAUTH_REDIRECT_BASE` to the exact local base URL if needed, and register the matching
callback URI with both providers. Disconnecting an account
removes its saved refresh token from the operating-system credential store.

## 4. Run

**Windows**

```powershell
.venv\Scripts\python -m app
```

**macOS / Linux**

```bash
.venv/bin/python -m app
```

You'll see:

```
  JobPilot AI -> http://127.0.0.1:8000
```

Open **http://127.0.0.1:8000** in your browser. Leave the terminal open while you use the app —
closing it (or pressing `Ctrl+C`) stops the server.

## 5. First use

1. **Upload your resume** (PDF, DOCX or TXT) on the *Matches* tab. With a Gemini key this takes up
   to a minute. Check the detected roles, skills and search queries — click **Edit** to adjust them.
   - Your resume is **analysed once** and the original file is kept in `data/resume/`. Every search
     (manual or daily) reuses the saved profile — the AI does not read your resume again.
   - Uploading the **same file again** is recognised and keeps your saved profile and edits.
     Uploading a **different** resume replaces the old one and analyses it.
   - **Re-analyse** (on the profile card) re-runs the analysis on demand — e.g. if Gemini was
     unavailable the first time. It overwrites your manual edits, so it asks you to click twice.
2. The **first search starts automatically** about 30 seconds after uploading (it takes 20–60 s).
   You can also press **Search now** at any time.
3. Work through the matches:
   - **★ Save** / **✓ Applied** — adds the job to the *Tracker* tab
   - **✕ Not interested** — hides the job from all future searches
   - **✉ Cover letter & tips** — tailored cover letter, resume tips and interview talking points
   - **📄 Tailor resume**: the AI rewrites your saved original resume for this job's description.
     Paste the full job description if the stored one is short, then click *Prepare*. You get an
     editable one-page preview: click any text to change it, press Enter to add a bullet, and use ✕ to remove one.
     The side panel shows JD keyword coverage (original → tailored), gaps the AI did **not** add,
     and warnings for any skill or figure that isn't in your original resume. Nothing is created until you
     click **Generate PDF** or **Generate DOCX**. Files are named `Company_Title_Resume.pdf` and are
     also kept in `data/tailored_resumes/` (linked from the Tracker).
   - **🏢 Company** — reviews, rating, typical pay and interview questions for the employer, plus
     one-click links to its AmbitionBox, Glassdoor, salary and LinkedIn pages. Uses only Gemini
     (no job-portal credits); results are cached per company (30 days, or 7 for AI estimates).
     If your Gemini key can use **Google Search** the data is live with sources ("Live Google
     data"); otherwise it's labelled **"AI estimate — not checked against live data"**.
   - **Export Excel / CSV** — download the results
4. Use the *Tracker* tab to move applications through interviewing / offer / rejected and keep notes.

## 6. The daily search

- Runs **every day at 08:00 IST** while the server is running — no setup needed.
- **Started the server after 08:00?** That day's search runs ~20 seconds after startup, so you never
  miss a day. If the server was off for several days, it runs once (not once per missed day).
- Change the time, location or turn it off on the **Schedule & sources** tab. **Run now** starts it
  immediately.
- New jobs are marked **NEW**; past runs are listed on the *History* tab (click one to reopen it).
- **See several days at once:** on the *Matches* tab, use **Show → Today / Last 3 / 7 / 14 days —
  combined**. All saved searches in that period (India calendar days) are merged into one list with
  duplicates removed — each job appears once with its latest score, "Seen in N searches", when it
  was first seen, and every apply link. **Export Excel / CSV** works on the combined list too.
- History keeps the **last 20 searches** (about 20 days of daily runs); jobs in the *Tracker* are kept
  permanently.

**Quota:** one search uses 3 JSearch calls and 3 SerpAPI calls — a daily search uses about 93 of each
per month, within the free tiers. The app reads the remaining quota from the providers (shown in the
header and on *Schedule & sources*) and skips a source when its quota is used up instead of failing.

## 7. Run the tests

```powershell
.venv\Scripts\python -m pytest -q
```

Tests never use your real API keys or your `data/` folder.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Install Python 3.11+ from https://www.python.org and tick *Add to PATH* |
| `ModuleNotFoundError` when starting | You're not using the project's Python — run `.venv\Scripts\python -m app`, not `python -m app` |
| Port 8000 already in use | Set `PORT=8001` in `.env` (or stop the other program) |
| "AI scoring off" in the header | `GEMINI_API_KEY` is missing in `.env`; restart the server after adding it |
| Resume upload slow (~40 s) | Gemini's main model is busy; the app falls back to another model automatically |
| "Gemini API key rejected" | The key is wrong or was deleted — create a new one and update `.env` |
| A source shows *skipped: quota used up* | Its free monthly quota is exhausted; it resumes after the reset date shown |
| Very few Chennai jobs | Add `RAPIDAPI_KEY` (JSearch) and `SERPAPI_KEY` (Google Jobs) |
| "Couldn't find readable text" on upload | The PDF is a scanned image — export your resume as a text PDF or DOCX |
| Changes to `.env` have no effect | Restart the server — settings are read at startup |
| Company info always says "AI estimate" | Your Gemini key can't use Google Search (not included on the free plan of this key). Enabling billing on the Google AI Studio project usually enables it — the app switches to live data automatically; press **Refresh** on a company to update |

## Where things are

```
app/
  __main__.py        starts the server (python -m app)
  main.py            web API routes
  config.py          settings, Chennai area names, company career boards
  sources/           job sources (free, company boards, JSearch/SerpAPI/Adzuna, quota guard)
  services/          resume parsing, filtering, scoring, storage, scheduler, export, cover letters
static/index.html    the web page
data/                your saved profile, resume file (data/resume/), searches, tracker
                     (created on first run; not in git)
tests/               automated tests
```
