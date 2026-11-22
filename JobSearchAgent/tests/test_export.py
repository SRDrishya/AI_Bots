from __future__ import annotations

import csv
import io

import pytest
from openpyxl import load_workbook

from app.models import JobLink, ResumeProfile, ScoredJob, StatusChange, TrackedJob
from app.services.coaching import _KitOut, generate_kit
from app.services import coaching
from app.services.export import jobs_csv, jobs_xlsx, safe, tracker_csv, tracker_xlsx

EVIL = '=HYPERLINK("http://evil.example","click")'


def job(**kw) -> ScoredJob:
    base = dict(id="jsearch:1", title="Data Engineer", company="Zoho", location="Chennai, Tamil Nadu",
                url="https://zoho.com/jobs/1", source="jsearch", match_score=88, location_match="local",
                salary="INR 12.0L - 18.0L / year", posted_at="2026-11-20T00:00:00+00:00", is_new=True,
                matched_skills=["Python", "Spark"], missing_skills=["dbt"], match_reasoning="Strong fit – Chennai ₹",
                alt_links=[JobLink(source="serpapi", url="https://linkedin.com/x")], key="k1")
    base.update(kw)
    return ScoredJob(**base)


@pytest.mark.parametrize("value", ["=1+1", "+91 98765", "-2", "@SUM(A1)", "\t=x", EVIL])
def test_safe_neutralises_formulas(value):
    assert safe(value) == "'" + value


@pytest.mark.parametrize("value", ["Data Engineer", "https://x.com", 88, None, ""])
def test_safe_leaves_normal_values(value):
    assert safe(value) == value


def test_jobs_csv():
    data = jobs_csv([job(), job(id="x:2", title=EVIL, company="+Evil Corp")])
    assert data.startswith("﻿".encode())  # BOM for Excel
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    header, first, evil = rows
    row = dict(zip(header, first))
    assert row["Score"] == "88" and row["Where"] == "Chennai" and row["Posted"] == "2026-11-20"
    assert row["Matched skills"] == "Python, Spark" and row["Other links"] == "https://linkedin.com/x"
    assert row["Why it fits"] == "Strong fit – Chennai ₹"  # unicode intact
    bad = dict(zip(header, evil))
    assert bad["Title"].startswith("'=") and bad["Company"] == "'+Evil Corp"


def test_jobs_xlsx_structure_and_safety():
    profile = ResumeProfile(candidate_name="Priya", skills=["Python"], summary="=cmd|' /C calc'!A0")
    content = jobs_xlsx([job(), job(id="x:2", title=EVIL, match_score=40)], profile, {"Location": "Chennai"})
    wb = load_workbook(io.BytesIO(content))
    assert wb.sheetnames == ["Matches", "Profile & search"]
    ws = wb["Matches"]
    headers = [c.value for c in ws[1]]
    col = {h: i + 1 for i, h in enumerate(headers)}
    assert ws.cell(2, col["Score"]).value == 88
    link = ws.cell(2, col["Apply link"])
    assert link.hyperlink.target == "https://zoho.com/jobs/1"
    evil = ws.cell(3, col["Title"])
    assert evil.data_type == "s" and evil.value.startswith("'=")  # stored as text, not a formula
    assert ws.freeze_panes == "C2" and ws.auto_filter.ref
    assert len(ws.conditional_formatting) >= 1
    prof = {r[0].value: r[1].value for r in wb["Profile & search"].iter_rows()}
    assert prof["Candidate"] == "Priya" and prof["Location"] == "Chennai"
    assert prof["Summary"].startswith("'=")


def test_empty_exports_are_valid():
    assert load_workbook(io.BytesIO(jobs_xlsx([])))["Matches"].max_row == 1
    assert load_workbook(io.BytesIO(tracker_xlsx([])))["Applications"].max_row == 1
    assert tracker_csv([]).decode("utf-8-sig").startswith("Status,Title")


def test_tracker_exports():
    item = TrackedJob(key="k1", job=job(), status="interviewing", notes="HR call Monday",
                      created_at="2026-11-20T10:00:00+00:00", updated_at="2026-11-22T10:00:00+00:00",
                      history=[StatusChange(status="applied", at="2026-11-20T10:00:00+00:00"),
                               StatusChange(status="interviewing", at="2026-11-22T10:00:00+00:00")])
    rows = list(csv.reader(io.StringIO(tracker_csv([item]).decode("utf-8-sig"))))
    row = dict(zip(*rows))
    assert row["Status"] == "interviewing" and row["Notes"] == "HR call Monday"
    assert row["History"] == "applied 2026-11-20 -> interviewing 2026-11-22"
    ws = load_workbook(io.BytesIO(tracker_xlsx([item])))["Applications"]
    assert ws.cell(2, 1).value == "interviewing"


# ------------------------------------------------------------------ application kit
def test_clean_letter_repairs_space_separated_paragraphs():
    from app.services.coaching import _clean_letter

    raw = ("Dear Hiring Manager,  I am writing to apply.  In my role at Freshworks I built pipelines.  "
           "Thank you for considering my application.  Sincerely, Priya Ramesh ")
    assert _clean_letter(raw) == (
        "Dear Hiring Manager,\n\nI am writing to apply.\n\nIn my role at Freshworks I built pipelines.\n\n"
        "Thank you for considering my application.\n\nSincerely,\nPriya Ramesh"
    )
    # Already well-formed letters are left alone (single spaces inside sentences untouched).
    good = "Dear Ms. Rao,\n\nI am keen to join.\n\nBest regards,\nPriya"
    assert _clean_letter(good) == good
async def test_generate_kit_prompt_and_cleanup(monkeypatch):
    seen = {}

    async def fake(*, prompt, schema, temperature, **_):
        seen["prompt"], seen["temperature"] = prompt, temperature
        return _KitOut(
            cover_letter="Dear Hiring Manager,\r\n\r\n\r\nI am **excited** to apply.\n\nRegards,\nPriya",
            resume_tips=["- Lead with Spark work", "1. Quantify pipeline volume", "Lead with Spark work", ""],
            keywords=["Python", "Spark"],
            talking_points=["• Address dbt honestly"],
        )

    monkeypatch.setattr(coaching, "generate_json", fake)
    profile = ResumeProfile(candidate_name="Priya Ramesh", skills=["Python"], years_of_experience=5, seniority="Mid")
    kit = await generate_kit(job(), profile, "FULL RESUME TEXT HERE", api_key="k", model="m", tone="warm")

    assert "FULL RESUME TEXT HERE" in seen["prompt"]
    assert "likely gaps: dbt" in seen["prompt"] and "Priya Ramesh" in seen["prompt"]
    assert "warm and personable" in seen["prompt"] and "Never invent employers" in seen["prompt"]
    assert kit.cover_letter == "Dear Hiring Manager,\n\nI am excited to apply.\n\nRegards,\nPriya"
    assert kit.resume_tips == ["Lead with Spark work", "Quantify pipeline volume"]
    assert kit.talking_points == ["Address dbt honestly"]
    assert kit.job_key == "k1" and kit.tone == "warm" and kit.created_at
