# InterviewBot

InterviewBot is a browser-based voice interview practice app that uses
OpenAI GPT-Live for voice conversation and WebRTC, and delegates interview
reasoning to GPT-6 Luna. The interviewer first asks what topic the candidate
wants to discuss and their preferred level. Luna uses the conversation and
each answer to create the next contextual interview question. It searches the
local document collection first and judges whether retrieved passages
adequately cover the topic and current question. It uses those documents when
sufficient and researches the web only when the local material is missing or
inadequate. Newly researched notes are saved as a PDF in `documents/` and as a
cited Markdown file in `topic_notes/`. GPT-Live speaks the exchange and
handles turn-taking and interruptions. The app does not score answers or make
hiring decisions.

## Features

- Natural spoken conversation with one interview question at a time.
- Uses GPT-Live's natural turn-taking and interruption handling, with
  interview planning delegated to GPT-6 Luna.
- Pause and resume the interview without ending the session.
- Displays live interviewer and candidate transcripts in the browser.
- Searches local PDF, DOCX, Markdown, and TXT documents before using web search.
- Saves newly researched topic notes as a PDF in `documents/` and a Markdown
  copy in `topic_notes/`, then retrieves their passages in later interviews.
- Keeps the conversation context during the session so follow-up questions can
  reflect your answers.
- Keeps the API key on the Python server; the browser never receives it.

## Requirements

- Python 3.10 or later.
- An OpenAI API key with access to GPT-Live and the Responses API/model
  configured for reasoning.
- A modern browser with microphone permission. Headphones are recommended to
  avoid audio feedback.
- Internet access.

## Install and run (PowerShell)

From this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Add your API key to `.env`, then start the local web server:

```powershell
interview-bot
```

Open `http://127.0.0.1:8000` in your browser and select **Start interview**.
The browser asks for microphone access after you click the button.

Alternatively, set the key in the current PowerShell session and run the
server:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
python main.py
```

Never commit `.env` or paste your API key into source files. The application
does not override environment variables already set in your shell. The server
binds to localhost and is intended for local development; add authentication
and HTTPS before exposing it to other users.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required | API key used by the Python server |
| `OPENAI_LIVE_MODEL` | `gpt-live-1` | GPT-Live model |
| `OPENAI_LIVE_VOICE` | `marin` | Generated interviewer voice |
| `OPENAI_REASONING_MODEL` | `gpt-6-luna` | Responses model that plans interview answers and next questions |
| `INTERVIEWER_INSTRUCTIONS` | Built-in prompt | Optional instructions for the spoken GPT-Live layer |
| `REASONING_INSTRUCTIONS` | Built-in prompt | Optional instructions for Luna's interview planning |

## Project layout

```text
InterviewBot/
├── .env.example
├── pyproject.toml
├── README.md
├── main.py
├── src/
│   └── interview_bot/
│       ├── config.py
│       ├── prompts.py
│       └── server.py
├── web/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── documents/            # Add local PDF, DOCX, Markdown, or TXT reference files
├── topic_notes/          # Generated Markdown notes and citations
└── tests/
```

Files in `documents/` are local to your machine and ignored by Git. Put
reference material there before starting an interview. The agent searches
these files and previously generated notes first; it calls web search only
when it judges that the available passages cannot support the discussion.
New web-researched topic briefs are saved both as a PDF in `documents/` and
as a cited Markdown file in `topic_notes/`.

## Tests

Run unit tests and lint from this folder:

```powershell
python -m pytest
ruff check .
```

Tests mock the Live API and do not require an API key or microphone.
