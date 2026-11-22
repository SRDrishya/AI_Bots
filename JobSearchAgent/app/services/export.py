"""CSV / Excel export of search results and the application tracker.

Job text comes from the internet, so every text cell is neutralised against
formula injection ("CSV injection"): a value starting with = + - @ (or tab/CR)
is prefixed with an apostrophe, which spreadsheet apps display as plain text.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime

from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from app.models import ResumeProfile, ScoredJob, TrackedJob

_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def safe(value) -> object:
    if isinstance(value, str) and value.startswith(_FORMULA_START):
        return "'" + value
    return value


def _date(iso: str) -> str:
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d")
    except ValueError:
        return iso


# (header, width, getter)
JOB_COLUMNS = [
    ("Score", 8, lambda j: j.match_score),
    ("Title", 42, lambda j: j.title),
    ("Company", 26, lambda j: j.company),
    ("Location", 28, lambda j: j.location),
    ("Where", 9, lambda j: {"local": "Chennai", "remote": "Remote"}.get(j.location_match, j.location_match)),
    ("Salary", 20, lambda j: j.salary),
    ("Type", 13, lambda j: j.employment_type),
    ("Posted", 11, lambda j: _date(j.posted_at)),
    ("New", 6, lambda j: "yes" if j.is_new else ""),
    ("First seen", 11, lambda j: _date(j.first_seen)),
    ("Times seen", 8, lambda j: j.appearances or ""),
    ("Status", 12, lambda j: j.tracking_status),
    ("Why it fits", 70, lambda j: j.match_reasoning),
    ("Matched skills", 40, lambda j: ", ".join(j.matched_skills)),
    ("Missing skills", 30, lambda j: ", ".join(j.missing_skills)),
    ("Apply link", 40, lambda j: j.url),
    ("Other links", 40, lambda j: " | ".join(link.url for link in j.alt_links)),
    ("Source", 11, lambda j: j.source),
    ("Scored by", 10, lambda j: j.scoring_engine),
]
TRACKER_COLUMNS = [
    ("Status", 12, lambda t: t.status),
    ("Title", 42, lambda t: t.job.title),
    ("Company", 26, lambda t: t.job.company),
    ("Location", 28, lambda t: t.job.location),
    ("Score", 8, lambda t: t.job.match_score),
    ("Salary", 20, lambda t: t.job.salary),
    ("Notes", 50, lambda t: t.notes),
    ("History", 50, lambda t: " -> ".join(f"{h.status} {_date(h.at)}" for h in t.history)),
    ("Added", 11, lambda t: _date(t.created_at)),
    ("Updated", 11, lambda t: _date(t.updated_at)),
    ("Link", 40, lambda t: t.job.url),
]


# --------------------------------------------------------------------------- CSV
def _csv(columns, rows) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([c[0] for c in columns])
    for row in rows:
        writer.writerow([safe(getter(row)) for _, _, getter in columns])
    return ("﻿" + buf.getvalue()).encode("utf-8")  # BOM so Excel opens UTF-8 (₹, –) correctly


def jobs_csv(jobs: list[ScoredJob]) -> bytes:
    return _csv(JOB_COLUMNS, jobs)


def tracker_csv(items: list[TrackedJob]) -> bytes:
    return _csv(TRACKER_COLUMNS, items)


# --------------------------------------------------------------------------- Excel
_HEADER_FILL = PatternFill("solid", fgColor="1F3A68")
_HEADER_FONT = Font(bold=True, color="FFFFFF")
_GOOD = PatternFill("solid", fgColor="D9F2E3")
_MID = PatternFill("solid", fgColor="E3EBFB")
_LINK = Font(color="1F5FCC", underline="single")
_WRAP = Alignment(wrap_text=True, vertical="top")
_TOP = Alignment(vertical="top")


def _sheet(wb: Workbook, title: str, columns, rows, *, link_columns=(), score_column=None, first=False):
    ws = wb.active if first else wb.create_sheet()
    ws.title = title
    ws.append([c[0] for c in columns])
    for cell in ws[1]:
        cell.fill, cell.font = _HEADER_FILL, _HEADER_FONT
        cell.alignment = Alignment(vertical="center")
    for row in rows:
        ws.append([safe(getter(row)) for _, _, getter in columns])

    for idx, (header, width, _) in enumerate(columns, start=1):
        letter = get_column_letter(idx)
        ws.column_dimensions[letter].width = width
        wrap = width >= 40 and header not in link_columns
        for cell in ws[letter][1:]:
            cell.alignment = _WRAP if wrap else _TOP
            if header in link_columns and isinstance(cell.value, str) and cell.value.startswith(("http://", "https://")):
                cell.hyperlink = cell.value
                cell.font = _LINK

    ws.freeze_panes = "C2"
    if ws.max_row > 1:
        ws.auto_filter.ref = ws.dimensions
        if score_column:
            col = get_column_letter([c[0] for c in columns].index(score_column) + 1)
            rng = f"{col}2:{col}{ws.max_row}"
            ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThanOrEqual", formula=["75"], fill=_GOOD))
            ws.conditional_formatting.add(rng, CellIsRule(operator="between", formula=["55", "74"], fill=_MID))
    return ws


def _profile_sheet(wb: Workbook, profile: ResumeProfile, meta: dict[str, str]) -> None:
    ws = wb.create_sheet("Profile & search")
    rows = [
        ("Exported", datetime.now().strftime("%Y-%m-%d %H:%M")),
        *meta.items(),
        ("", ""),
        ("Candidate", profile.candidate_name),
        ("Seniority", profile.seniority),
        ("Experience (years)", profile.years_of_experience if profile.years_of_experience is not None else ""),
        ("Target roles", ", ".join(profile.target_roles)),
        ("Search queries", ", ".join(profile.search_queries)),
        ("Skills", ", ".join(profile.skills)),
        ("Summary", profile.summary),
    ]
    for key, value in rows:
        ws.append([safe(key), safe(value)])
        ws.cell(ws.max_row, 1).font = Font(bold=True)
        ws.cell(ws.max_row, 2).alignment = _WRAP
    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 100


def _to_bytes(wb: Workbook) -> bytes:
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def jobs_xlsx(jobs: list[ScoredJob], profile: ResumeProfile | None = None, meta: dict[str, str] | None = None) -> bytes:
    wb = Workbook()
    _sheet(wb, "Matches", JOB_COLUMNS, jobs, link_columns=("Apply link",), score_column="Score", first=True)
    if profile is not None:
        _profile_sheet(wb, profile, meta or {})
    return _to_bytes(wb)


def tracker_xlsx(items: list[TrackedJob]) -> bytes:
    wb = Workbook()
    _sheet(wb, "Applications", TRACKER_COLUMNS, items, link_columns=("Link",), score_column="Score", first=True)
    return _to_bytes(wb)
