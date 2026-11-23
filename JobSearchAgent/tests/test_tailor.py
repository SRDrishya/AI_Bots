from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app import main as main_mod
from app.models import (
    ResumeContact,
    ResumeDocument,
    ResumeEducation,
    ResumeProject,
    ResumeRole,
    ResumeSkillGroup,
    TailoredResume,
)
from app.services import tailor
from app.services.resume_render import fit_pdf, render_docx
from app.services.storage import Storage
from tests.test_resume import SAMPLE


def make_doc(roles: int = 2, bullets: int = 3) -> ResumeDocument:
    return ResumeDocument(
        contact=ResumeContact(name="Priya Ramesh", headline="Data Engineer", email="priya.r@example.com",
                              phone="+91 98765 43210", location="Chennai", links=["https://linkedin.com/in/priya"]),
        summary="Senior Data Engineer building batch and streaming pipelines on AWS with Python, PySpark and Airflow.",
        skills=[ResumeSkillGroup(category="Languages", items=["Python", "SQL"]),
                ResumeSkillGroup(category="Data", items=["Apache Spark", "Airflow", "Kafka"])],
        experience=[ResumeRole(title="Senior Data Engineer", company=f"Company {i}", location="Chennai",
                               dates="2020 – Present", bullets=["Built ETL pipelines with Spark and Airflow on AWS S3"] * bullets)
                    for i in range(roles)],
        projects=[ResumeProject(name="Streaming alerts", tech="Kafka", bullets=["Streamed events to alerting"])],
        education=[ResumeEducation(degree="B.E. Computer Science", institution="Anna University", dates="2013 – 2017")],
    )


# --------------------------------------------------------------------------- rendering
def test_pdf_fits_one_page_and_shrinks_when_needed():
    from pypdf import PdfReader

    data, fit = fit_pdf(make_doc())
    assert fit.fits and fit.pages == 1 and fit.font_size == 10.5
    text = PdfReader(io.BytesIO(data)).pages[0].extract_text()
    assert "Priya Ramesh" in text and "EXPERIENCE" in text

    _, bigger = fit_pdf(make_doc(roles=7, bullets=6))
    assert bigger.fits and bigger.font_size < 10.5

    _, huge = fit_pdf(make_doc(roles=12, bullets=8))
    assert not huge.fits and huge.pages >= 2


def test_docx_contains_all_sections():
    import docx

    d = docx.Document(io.BytesIO(render_docx(make_doc(), 10)))
    text = "\n".join(p.text for p in d.paragraphs)
    for expected in ("Priya Ramesh", "PROFESSIONAL SUMMARY", "SKILLS", "EXPERIENCE", "PROJECTS", "EDUCATION",
                     "Languages: Python, SQL", "2020 – Present"):
        assert expected in text


def test_projects_first_for_freshers():
    from app.services.resume_render import _sections

    doc = make_doc()
    assert _sections(doc).index("experience") < _sections(doc).index("projects")
    doc.experience = []
    assert "experience" not in _sections(doc)
    doc.experience = [ResumeRole(title="Intern", company="X")]
    assert _sections(doc).index("projects") < _sections(doc).index("experience")


# --------------------------------------------------------------------------- checks
def test_honesty_warnings_flag_new_skills_and_figures():
    doc = make_doc()
    assert tailor.honesty_warnings(SAMPLE, doc) == []

    doc.skills[0].items.append("Rust")
    doc.experience[0].bullets[0] = "Cut pipeline cost by 35% for 12 clients"
    warnings = tailor.honesty_warnings(SAMPLE, doc)
    assert any("Rust" in w for w in warnings)
    assert any("35%" in w and "12" in w for w in warnings)


def test_coverage_counts_original_vs_tailored():
    doc = make_doc()
    cov = tailor.coverage(["Python", "Kafka", "Terraform", "AWS"], "Python developer", doc)
    hits = {k.term: (k.in_original, k.in_tailored) for k in cov.keywords}
    assert hits["Python"] == (True, True)
    assert hits["Kafka"] == (False, True)
    assert hits["Terraform"] == (False, False)
    assert (cov.before, cov.after, cov.total) == (1, 3, 4)


def test_tidy_and_file_stem():
    doc = make_doc()
    doc.experience[0].bullets = ["• Built things", "", "  - Shipped things  "]
    doc.skills.append(ResumeSkillGroup(category="Empty", items=[]))
    clean = tailor.tidy(doc)
    assert clean.experience[0].bullets == ["Built things", "Shipped things"]
    assert all(g.items for g in clean.skills)
    assert tailor.file_stem("Zoho Corp.", "Sr. Python Developer (Remote)") == "Zoho_Corp_Sr_Python_Developer_Remote_Resume"


# --------------------------------------------------------------------------- API
@pytest.fixture
def client(tmp_path):
    app = main_mod.create_app(Storage(tmp_path / "data"), start_scheduler=False)
    with TestClient(app) as c:
        yield c


JOB = {"id": "jsearch:1", "title": "Data Engineer", "company": "Zoho", "url": "https://z/1",
       "source": "jsearch", "match_score": 80, "description": "Python, Spark, Kafka and Terraform on AWS."}


def test_tailored_resume_flow(client, monkeypatch):
    r = client.post("/api/tailored-resume", json={"job": JOB})
    assert r.status_code == 400 and "Gemini" in r.json()["detail"]

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert client.post("/api/tailored-resume", json={"job": JOB}).status_code == 400  # no resume yet
    client.post("/api/resume", files={"file": ("cv.txt", SAMPLE.encode(), "text/plain")})

    calls = []

    async def fake_prepare(job, description, profile, resume_text, *, api_key, model):
        calls.append(description)
        draft = TailoredResume(job_key=job.key, job_title=job.title, company=job.company,
                               job_description=description, resume=make_doc(), gaps=["Terraform"],
                               jd_keywords=["Python", "Kafka", "Terraform"], created_at="2026-11-23T10:00:00+00:00")
        return tailor.refresh(draft, resume_text)

    monkeypatch.setattr(main_mod, "prepare_tailored", fake_prepare)
    draft = client.post("/api/tailored-resume", json={"job": JOB, "job_description": "Full pasted JD"}).json()
    key = draft["job_key"]
    assert calls == ["Full pasted JD"] and draft["fit"]["fits"] and draft["coverage"]["total"] == 3

    # cached unless regenerate; empty pasted JD falls back to the job's own description
    assert client.post("/api/tailored-resume", json={"job": JOB}).json()["job_key"] == key and len(calls) == 1
    client.post("/api/tailored-resume", json={"job": JOB, "regenerate": True})
    assert calls[-1] == JOB["description"]

    # edits from the preview are saved and re-checked
    edited = draft["resume"]
    edited["skills"][0]["items"].append("Rust")
    saved = client.put(f"/api/tailored-resume/{key}", json=edited).json()
    assert "Rust" in saved["resume"]["skills"][0]["items"]
    assert any("Rust" in w for w in saved["warnings"])
    assert client.get(f"/api/tailored-resume/{key}").json()["resume"]["skills"][0]["items"][-1] == "Rust"

    # nothing generated until Generate is clicked
    assert client.get(f"/api/tailored-resume/{key}").json()["files"] == []

    pdf = client.post(f"/api/tailored-resume/{key}/generate?format=pdf", json=edited)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert 'filename="Zoho_Data_Engineer_Resume.pdf"' in pdf.headers["content-disposition"]
    assert pdf.headers["x-page-count"] == "1"
    docx_res = client.post(f"/api/tailored-resume/{key}/generate?format=docx", json=edited)
    assert docx_res.status_code == 200 and docx_res.content[:2] == b"PK"
    assert client.post(f"/api/tailored-resume/{key}/generate?format=txt", json=edited).status_code == 400

    files = client.get(f"/api/tailored-resume/{key}").json()["files"]
    assert {f["filename"] for f in files} == {"Zoho_Data_Engineer_Resume.pdf", "Zoho_Data_Engineer_Resume.docx"}
    again = client.get(f"/api/tailored-resume/{key}/files/Zoho_Data_Engineer_Resume.pdf")
    assert again.status_code == 200 and again.content == pdf.content
    assert client.get(f"/api/tailored-resume/{key}/files/..%2Fprofile.json").status_code == 404
    assert [d["job_key"] for d in client.get("/api/tailored-resumes").json()] == [key]

    # re-preparing keeps already generated files downloadable
    client.post("/api/tailored-resume", json={"job": JOB, "regenerate": True})
    assert len(client.get(f"/api/tailored-resume/{key}").json()["files"]) == 2


def test_tailored_resume_unknown_key(client):
    assert client.get("/api/tailored-resume/abc123").status_code == 404
    assert client.get("/api/tailored-resume/bad key!").status_code == 404
    assert client.put("/api/tailored-resume/abc123", json={}).status_code == 404
