import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.models import Answer, Bank, Evaluation, Question, Source
from app.services import catalog, export


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr("app.services.storage.JOBSEARCH_RESUME_FILE", tmp_path / "missing.json")
    from app.main import app

    with TestClient(app) as c:
        yield c


def sample_bank(slug="csharp", exp="5-7"):
    key = f"{slug}--{exp}"
    return Bank(key=key, slug=slug, name="C#", experience=exp, categories=["Basics", "Async"], questions=[
        Question(id=f"{key}-1", question="What is a delegate in C#?", category="Basics", level="beginner",
                 sources=[Source(title="Docs", url="https://learn.microsoft.com")],
                 answer=Answer(summary="A type-safe function pointer.",
                               explanation="Para one.\n\n- bullet `code` **bold**\n\n```\nx\n```",
                               code_language="csharp", code_example="Func<int,int> f = x => x * 2; // ✓ unicode",
                               key_points=["Multicast"], pitfalls=["Leaks"], follow_ups=["Events?"],
                               your_angle="Used in policy routing.")),
        Question(id=f"{key}-2", question="How does ConfigureAwait(false) work?", category="Async",
                 level="advanced"),
    ])


def test_status_lists_experience_levels(client):
    s = client.get("/api/status").json()
    assert s["default_experience"] == "5-7"
    assert [e["key"] for e in s["experiences"]] == ["0-2", "3-4", "5-7", "8-10", "11-plus"]
    assert s["semantic"] is False  # no Gemini key in tests


def test_technologies_include_resume_skills_and_rounds(client):
    names = {t["name"] for t in client.get("/api/technologies?exp=5-7").json()}
    assert {"C#", "ADO.NET", "Entity Framework (EF Core & EF6)", "LINQ", "SQL Server", "System Design & Microservices",
            "Design Patterns & SOLID", "Behavioral & Leadership"} <= names
    assert "My Resume Projects" not in names


def test_counts_are_per_experience(client):
    client.app.state.storage.save_bank(sample_bank())
    by_exp = {e: {t["slug"]: t["count"] for t in client.get(f"/api/technologies?exp={e}").json()}
              for e in ("5-7", "0-2")}
    assert by_exp["5-7"]["csharp"] == 2 and by_exp["0-2"]["csharp"] == 0
    assert client.get("/api/technologies?exp=99").status_code == 400


def test_add_and_remove_custom_technology(client):
    tech = client.post("/api/technologies", json={"name": "Docker", "kind": "tech"}).json()
    assert tech["slug"] == "docker"
    assert client.post("/api/technologies", json={"name": "docker"}).json()["slug"] == "docker"
    assert client.delete("/api/technologies/docker").status_code == 200
    assert client.delete("/api/technologies/csharp").status_code == 400


def test_bank_progress_and_text_search(client):
    client.app.state.storage.save_bank(sample_bank())
    assert len(client.get("/api/banks/csharp?exp=5-7").json()["bank"]["questions"]) == 2
    assert client.get("/api/banks/csharp?exp=0-2").json()["bank"] is None
    assert client.put("/api/progress/csharp--5-7-1", json={"status": "known"}).json()["status"] == "known"
    r = client.get("/api/banks/csharp/search?exp=5-7&q=delegate").json()
    assert r["mode"] == "text" and [x["id"] for x in r["results"]] == ["csharp--5-7-1"]


@pytest.mark.parametrize("fmt", ["pdf", "docx", "html"])
def test_export_only_selected_questions(client, fmt):
    client.app.state.storage.save_bank(sample_bank())
    r = client.post("/api/banks/csharp/export?exp=5-7",
                    json={"format": fmt, "question_ids": ["csharp--5-7-1"], "include_sources": True})
    assert r.status_code == 200, r.text
    if fmt == "pdf":
        assert r.content.startswith(b"%PDF")
    elif fmt == "docx":
        xml = zipfile.ZipFile(io.BytesIO(r.content)).read("word/document.xml").decode()
        assert "What is a delegate in C#?" in xml and "ConfigureAwait" not in xml
    else:
        assert "What is a delegate in C#?" in r.text and "ConfigureAwait" not in r.text
        assert "5–7 years" in r.text and "learn.microsoft.com" in r.text


def test_export_without_angle_and_unknown_ids(client):
    client.app.state.storage.save_bank(sample_bank())
    r = client.post("/api/banks/csharp/export?exp=5-7",
                    json={"format": "html", "question_ids": ["csharp--5-7-1"], "include_angle": False})
    assert "policy routing" not in r.text
    bad = client.post("/api/banks/csharp/export?exp=5-7", json={"format": "html", "question_ids": ["nope"]})
    assert bad.status_code == 400


def test_evaluate_records_attempt(client, monkeypatch):
    client.app.state.storage.save_bank(sample_bank())

    async def fake_evaluate(tech_name, experience, question, answer, resume):
        assert experience.key == "5-7"
        return Evaluation(score=9, verdict="Strong", strengths=["clear"], gaps=[], improved_answer="")

    monkeypatch.setattr("app.main.practice.evaluate", fake_evaluate)
    monkeypatch.setattr("app.main.ai.ai_available", lambda: True)
    r = client.post("/api/practice/evaluate", json={"slug": "csharp", "experience": "5-7",
                                                    "question_id": "csharp--5-7-1", "answer": "A pointer"})
    assert r.status_code == 200
    assert r.json()["progress"]["status"] == "known"


def test_role_banks_and_target_companies(client):
    techs = {t["slug"]: t for t in client.get("/api/technologies?exp=5-7").json()}
    assert techs["lead-dotnet-fullstack"]["kind"] == "role"
    assert techs["dotnet-fullstack-genai"]["group"] == "Target roles"
    assert "Cognizant" in client.get("/api/status").json()["companies"]
    saved = client.put("/api/preferences", json={"companies": [" Zoho ", "TCS", "zoho", ""]}).json()
    assert saved["companies"] == ["Zoho", "TCS", "zoho"]
    assert client.get("/api/status").json()["companies"] == saved["companies"]


def test_export_shows_interview_evidence(client):
    bank = sample_bank()
    q = bank.questions[0]
    q.reported, q.companies = True, ["Infosys", "TCS"]
    q.sources.append(Source(title="AmbitionBox", url="https://www.ambitionbox.com/x"))
    client.app.state.storage.save_bank(bank)
    r = client.post("/api/banks/csharp/export?exp=5-7", json={"format": "html", "question_ids": [q.id]})
    assert "Reported by candidates at Infosys, TCS · found on 2 sites" in r.text


def test_build_requires_ai_key(client):
    assert client.post("/api/banks/csharp/build?exp=5-7").status_code == 400


def test_markdown_blocks_and_ordering():
    bank = sample_bank()
    groups = export.ordered(bank, list(reversed(bank.questions)))
    assert [c for c, _ in groups] == ["Basics", "Async"]
    assert export.blocks("a\nb\n\n- x\n```\ncode\n```") == [("para", "a b"), ("bullet", "x"), ("code", "code")]


def test_slugify():
    assert catalog.slugify("C#") == "csharp"
