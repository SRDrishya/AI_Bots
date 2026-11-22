from __future__ import annotations

import io
from datetime import date

import pytest

from app.services import resume as resume_mod
from app.services.resume import (
    ResumeParseError,
    _GeminiResume,
    analyze_local,
    analyze_resume,
    extract_text,
)
from app.services.skills import extract_roles, extract_skills

SAMPLE = """PRIYA RAMESH
priya.r@example.com | +91 98765 43210 | Chennai, Tamil Nadu

Professional Summary
Senior Data Engineer with experience building batch and streaming pipelines
on AWS using Python, PySpark, Airflow and Kafka.

Experience
Senior Data Engineer, Freshworks — Chennai
2020 - Present
- Built ETL pipelines with Spark and Airflow; data lake on S3 with Snowflake.
Data Analyst, TCS — Chennai
2017 - 2020
- SQL reporting, Power BI dashboards, Python automation.

Education
B.E. Computer Science, Anna University, 2013 - 2017

Skills
Python, SQL, PySpark, Airflow, Kafka, AWS, Docker, Snowflake, Power BI, Git
"""


def _docx_bytes(text: str, table_rows: list[list[str]] | None = None) -> bytes:
    import docx

    document = docx.Document()
    for line in text.splitlines():
        document.add_paragraph(line)
    if table_rows:
        table = document.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r, row in enumerate(table_rows):
            for c, value in enumerate(row):
                table.cell(r, c).text = value
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------- extraction
def test_extract_txt():
    assert "PRIYA RAMESH" in extract_text("cv.txt", SAMPLE.encode())


def test_extract_docx_includes_tables():
    data = _docx_bytes(SAMPLE, table_rows=[["Cloud", "Kubernetes, Terraform"]])
    text = extract_text("cv.docx", data)
    assert "Freshworks" in text
    assert "Kubernetes" in text  # came from the table


def test_blank_pdf_reports_no_text():
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(ResumeParseError, match="scanned"):
        extract_text("cv.pdf", buf.getvalue())


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        ("cv.pdf", b"", "empty"),
        ("cv.doc", b"x" * 100, ".doc"),
        ("cv.png", b"x" * 100, "Unsupported"),
        ("cv.pdf", b"not a pdf at all" * 10, "corrupted"),
        ("cv.docx", b"not a docx" * 10, "corrupted"),
    ],
)
def test_bad_inputs(filename, data, message):
    with pytest.raises(ResumeParseError, match=message):
        extract_text(filename, data)


def test_file_too_large(monkeypatch):
    monkeypatch.setattr(resume_mod, "MAX_RESUME_BYTES", 10)
    with pytest.raises(ResumeParseError, match="larger"):
        extract_text("cv.txt", b"x" * 11)


# ---------------------------------------------------------------- skills
def test_skill_matching_boundaries():
    skills = extract_skills("JavaScript and C++ and C# with .NET; k8s; rest of the team")
    assert {"JavaScript", "C++", "C#", ".NET", "Kubernetes"} <= set(skills)
    assert "Java" not in skills
    assert "REST APIs" not in skills


def test_role_aliases():
    assert extract_roles("Worked as an SDE and later an ML Engineer") == [
        "Software Engineer",
        "Machine Learning Engineer",
    ]


# ---------------------------------------------------------------- local analysis
def test_analyze_local_profile():
    profile = analyze_local(extract_text("cv.txt", SAMPLE.encode()))
    assert profile.candidate_name == "Priya Ramesh"
    assert profile.analysis_engine == "local"
    assert profile.target_roles[0] == "Data Engineer"
    assert {"Python", "Spark", "Airflow", "Kafka", "AWS", "Snowflake"} <= set(profile.skills)
    # 2017 -> today from job ranges only; the education range must not count.
    assert profile.years_of_experience == float(date.today().year - 2017)
    assert profile.seniority in {"Senior", "Lead"}
    assert profile.preferred_locations[:2] == ["Chennai", "Remote"]
    assert "Data Engineer" in profile.search_queries
    assert "Senior Data Engineer" in profile.summary


def test_stated_years_wins():
    text = SAMPLE.replace("Senior Data Engineer with", "Senior Data Engineer with 6+ years of")
    assert analyze_local(text).years_of_experience == 6.0


# ---------------------------------------------------------------- gemini path
async def test_no_key_uses_local():
    profile = await analyze_resume("cv.txt", SAMPLE.encode())
    assert profile.analysis_engine == "local"


async def test_gemini_failure_falls_back(monkeypatch):
    async def boom(*_args, **_kwargs):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(resume_mod, "analyze_gemini", boom)
    profile = await analyze_resume("cv.txt", SAMPLE.encode(), gemini_api_key="fake")
    assert profile.analysis_engine == "local (gemini failed)"
    assert profile.skills


async def test_gemini_merge(monkeypatch):
    async def fake(*_args, **_kwargs):
        return _GeminiResume(
            candidate_name="Priya Ramesh",
            summary="Data engineer focused on streaming.",
            target_roles=["Senior Data Engineer", "Analytics Engineer"],
            skills=["pyspark", "Apache Flink"],
            years_of_experience=7.5,
            seniority="senior",
            current_location="Chennai",
            search_queries=[],
        )

    monkeypatch.setattr(resume_mod, "analyze_gemini", fake)
    profile = await analyze_resume("cv.txt", SAMPLE.encode(), gemini_api_key="fake")
    assert profile.analysis_engine == "gemini"
    assert profile.skills[:2] == ["Spark", "Apache Flink"]  # normalized + kept unknown skill
    assert "Kafka" in profile.skills  # local skills fill in
    assert profile.years_of_experience == 7.5
    assert profile.seniority == "Senior"
    assert profile.search_queries  # rebuilt because Gemini returned none
    assert profile.preferred_locations == ["Chennai", "Remote"]
