# Tutorial Generator

Tutorial Generator is a local browser app that researches a topic on the web
and creates a clear, beginner-friendly PDF tutorial. Its agent searches for
relevant documentation and tutorial sources, explains the main ideas in
simple English in a friendly, analogy-first lesson. It includes a practical
example, a visual flow diagram, and a short code or pseudocode example when
that helps explain a technical topic. Linked web sources are listed in the
PDF. Interview questions and answers are handled by a separate feature, not
included in the tutorial document.

The PDF title uses the topic entered in the form. The generator checks that
the lesson contains its required teaching sections, has at least 350 words
and 200 words of explanatory prose, includes inline source citations, and
contains the visual diagram before returning it.

The PDF is created only when web research returns verifiable HTTP(S) source
links and the agent includes its required diagram and valid inline citations.
If the first draft omits or misnumbers inline citations, the agent gets one
automatic rewrite attempt using the same research and verified source list.
The app reports a useful error rather than producing an unsourced or
incomplete tutorial.

## Requirements

- Python 3.10 or later.
- An OpenAI API key with access to the configured model and web search.
- Internet access.

## Install and run (PowerShell)

From this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Add your API key to `.env`, then start the app:

```powershell
tutorial-generator
```

Alternatively, run `python main.py`. Open `http://127.0.0.1:8001` in your
browser, enter a topic, and choose **Generate PDF tutorial**.

The API key stays on the Python server and is never sent to the browser.
The server binds to localhost for local use; add authentication and HTTPS
before exposing it to other users.

### Troubleshooting

If `python main.py` reports `No module named 'tutorial_generator'`, install
the project into the same virtual environment that runs it. From this folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe main.py
```

Using the virtual environment's Python for both commands avoids installing
the package into a different Python environment.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required | API key used for web research and tutorial generation |
| `TUTORIAL_MODEL` | `gpt-4.1-mini` | Model used by the tutorial agent and web research |

## Project layout

```text
Tutorial_Generator/
├── .env.example
├── pyproject.toml
├── README.md
├── main.py
├── src/
│   └── tutorial_generator/
│       ├── agent.py       # Research tool and tutorial-writing agent
│       ├── config.py
│       ├── document.py    # PDF layout, source links, and flow diagram
│       ├── prompts.py
│       ├── research.py    # Web search and citation validation
│       └── server.py
├── tests/
└── web/
```

## Tests and lint

Run these commands from this folder:

```powershell
python -m pytest
ruff check .
```
