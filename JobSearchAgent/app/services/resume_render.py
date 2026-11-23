"""Render a ResumeDocument to PDF (fpdf2) or DOCX (python-docx) with one clean, ATS-friendly template.

Single column, standard section names, real text (no images/tables/text boxes), so applicant
tracking systems can parse it. The PDF is shrunk step by step (font size + spacing) until it fits
on one page; the DOCX uses the same font size so both files look alike.
"""
from __future__ import annotations

import io
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

from app.models import PageFit, ResumeDocument

FONT_SIZES = (10.5, 10.0, 9.5, 9.0, 8.5)   # tried in order until the resume fits on one page
MARGIN_MM = 13.0
PT_TO_MM = 0.3528

# Unicode TTF families tried in order; the built-in Helvetica (Latin-1 only) is the last resort.
_FONT_FAMILIES = [
    ("Calibri", {"": "calibri.ttf", "B": "calibrib.ttf", "I": "calibrii.ttf"}),
    ("Arial", {"": "arial.ttf", "B": "arialbd.ttf", "I": "ariali.ttf"}),
    ("Carlito", {"": "Carlito-Regular.ttf", "B": "Carlito-Bold.ttf", "I": "Carlito-Italic.ttf"}),
    ("LiberationSans", {"": "LiberationSans-Regular.ttf", "B": "LiberationSans-Bold.ttf",
                        "I": "LiberationSans-Italic.ttf"}),
    ("DejaVuSans", {"": "DejaVuSans.ttf", "B": "DejaVuSans-Bold.ttf", "I": "DejaVuSans-Oblique.ttf"}),
]
_SECTION_TITLES = {
    "summary": "Professional Summary", "skills": "Skills", "experience": "Experience",
    "projects": "Projects", "education": "Education", "certifications": "Certifications",
    "achievements": "Achievements",
}


def _font_dirs() -> list[Path]:
    dirs = []
    if sys.platform == "win32":
        dirs.append(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts")
        if local := os.environ.get("LOCALAPPDATA"):
            dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    elif sys.platform == "darwin":
        dirs += [Path("/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"), Path.home() / "Library/Fonts"]
    else:
        dirs += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts"]
    return [d for d in dirs if d.is_dir()]


@lru_cache(maxsize=1)
def find_font() -> tuple[str, dict[str, Path]] | None:
    """(family, {style: path}) for the first installed Unicode font family, or None."""
    dirs = _font_dirs()
    for family, files in _FONT_FAMILIES:
        found: dict[str, Path] = {}
        for style, name in files.items():
            for d in dirs:
                hits = [d / name] if (d / name).is_file() else list(d.rglob(name))[:1] if sys.platform != "win32" else []
                if hits:
                    found[style] = hits[0]
                    break
        if len(found) == len(files):
            return family, found
    return None


def docx_font_name() -> str:
    font = find_font()
    return font[0] if font and font[0] in ("Calibri", "Arial") else "Calibri"


_ASCII = str.maketrans({"\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                        "\u2022": "-", "\u2026": "...", "\u00a0": " ", "\u20b9": "Rs.", "\u2192": "->"})


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _short_link(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url.strip()).rstrip("/")


def contact_line(doc: ResumeDocument) -> str:
    c = doc.contact
    return "  |  ".join(_clean(x) for x in [c.location, c.phone, c.email, *map(_short_link, c.links)] if _clean(x))


def _sections(doc: ResumeDocument) -> list[str]:
    """Section order: summary, skills, then experience before projects (projects first for freshers)."""
    order = ["summary", "skills", "experience", "projects", "education", "certifications", "achievements"]
    has_bullets = any(r.bullets for r in doc.experience)
    if not has_bullets and doc.projects:
        order[2], order[3] = "projects", "experience"
    present = {
        "summary": bool(_clean(doc.summary)),
        "skills": any(g.items for g in doc.skills),
        "experience": bool(doc.experience),
        "projects": bool(doc.projects),
        "education": bool(doc.education),
        "certifications": any(_clean(c) for c in doc.certifications),
        "achievements": any(_clean(a) for a in doc.achievements),
    }
    return [s for s in order if present[s]]


def _role_parts(r) -> tuple[list[tuple[str, str]], str]:
    parts = [(_clean(r.title), "B")]
    if _clean(r.company):
        parts.append((("  |  " if parts[0][0] else "") + _clean(r.company), ""))
    right = "  |  ".join(x for x in (_clean(r.location), _clean(r.dates)) if x)
    return parts, right


def _project_parts(p) -> tuple[list[tuple[str, str]], str]:
    parts = [(_clean(p.name), "B")]
    if _clean(p.tech):
        parts.append(("  |  " + _clean(p.tech), "I"))
    return parts, _clean(p.dates)


def _edu_parts(e) -> tuple[list[tuple[str, str]], str]:
    parts = [(_clean(e.degree), "B")]
    rest = ", ".join(x for x in (_clean(e.institution), _clean(e.location)) if x)
    if rest:
        parts.append((("  |  " if parts[0][0] else "") + rest, ""))
    if _clean(e.details):
        parts.append(("  |  " + _clean(e.details), ""))
    return parts, _clean(e.dates)


# --------------------------------------------------------------------------- PDF
class _Pdf:
    def __init__(self, size: float):
        from fpdf import FPDF

        self.size = size
        self.lh = size * PT_TO_MM * 1.22          # line height
        self.gap = size * 0.28                     # space before a section
        self.pdf = FPDF(format="A4", unit="mm")
        self.pdf.set_margins(MARGIN_MM, MARGIN_MM * 0.85, MARGIN_MM)
        self.pdf.set_auto_page_break(True, MARGIN_MM * 0.85)
        font = find_font()
        if font:
            self.family, self.unicode = "ResumeFont", True
            for style, path in font[1].items():
                self.pdf.add_font("ResumeFont", style, str(path))
        else:
            self.family, self.unicode = "Helvetica", False
        self.bullet = "\u2022" if self.unicode else "-"
        self.pdf.add_page()

    def t(self, text: str, strip: bool = True) -> str:
        text = _clean(text) if strip else re.sub(r"\s+", " ", text or "")
        if not self.unicode:
            text = text.translate(_ASCII).encode("latin-1", "replace").decode("latin-1")
        return text

    def font(self, style: str = "", delta: float = 0):
        self.pdf.set_font(self.family, style, self.size + delta)

    def centered(self, text: str, style: str = "", delta: float = 0):
        from fpdf.enums import XPos, YPos

        if not _clean(text):
            return
        self.font(style, delta)
        lh = (self.size + delta) * PT_TO_MM * 1.25
        self.pdf.multi_cell(0, lh, self.t(text), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def heading(self, title: str):
        p = self.pdf
        p.ln(self.gap)
        self.font("B", 1.5)
        p.cell(0, (self.size + 1.5) * PT_TO_MM * 1.2, self.t(title.upper()))
        p.ln((self.size + 1.5) * PT_TO_MM * 1.2)
        y = p.get_y() + 0.2
        p.set_line_width(0.25)
        p.line(p.l_margin, y, p.w - p.r_margin, y)
        p.ln(1.2)

    def line(self, parts: list[tuple[str, str]], right: str = ""):
        """Mixed-style text on the left (wraps), optional text right-aligned on the first line."""
        p = self.pdf
        if p.get_y() + self.lh > p.page_break_trigger:
            p.add_page()
        y, old_margin = p.get_y(), p.r_margin
        if right:
            self.font("")
            rw = p.get_string_width(self.t(right)) + 3
            p.set_xy(p.w - old_margin - rw, y)
            p.cell(rw, self.lh, self.t(right), align="R")
            p.set_xy(p.l_margin, y)
            p.set_right_margin(old_margin + rw)
        for text, style in parts:
            if text:
                self.font(style)
                p.write(self.lh, self.t(text, strip=False))
        p.set_right_margin(old_margin)
        p.ln(self.lh)

    def para(self, text: str):
        from fpdf.enums import XPos, YPos

        self.font("")
        self.pdf.multi_cell(0, self.lh, self.t(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def bullets(self, items: list[str]):
        from fpdf.enums import XPos, YPos

        p, indent = self.pdf, 4.0
        for item in filter(None, map(_clean, items)):
            self.font("")
            p.set_x(p.l_margin + 1.0)
            p.cell(indent - 1.0, self.lh, self.bullet)
            p.multi_cell(p.epw - indent, self.lh, self.t(item), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def entry_gap(self):
        self.pdf.ln(self.size * 0.09)


def render_pdf(doc: ResumeDocument, font_size: float = FONT_SIZES[0]) -> tuple[bytes, int]:
    w = _Pdf(font_size)
    w.centered(doc.contact.name or "Resume", "B", 9)
    w.centered(doc.contact.headline, "", 1)
    w.centered(contact_line(doc), "", -0.5)
    for section in _sections(doc):
        w.heading(_SECTION_TITLES[section])
        if section == "summary":
            w.para(doc.summary)
        elif section == "skills":
            for g in doc.skills:
                items = ", ".join(filter(None, map(_clean, g.items)))
                if items:
                    w.line([(f"{_clean(g.category)}: " if _clean(g.category) else "", "B"), (items, "")])
        elif section in ("experience", "projects", "education"):
            entries = {"experience": doc.experience, "projects": doc.projects, "education": doc.education}[section]
            parts_of = {"experience": _role_parts, "projects": _project_parts, "education": _edu_parts}[section]
            for i, entry in enumerate(entries):
                if i:
                    w.entry_gap()
                parts, right = parts_of(entry)
                w.line(parts, right)
                w.bullets(getattr(entry, "bullets", []))
        else:
            w.bullets(getattr(doc, section))
    return bytes(w.pdf.output()), w.pdf.pages_count


def fit_pdf(doc: ResumeDocument) -> tuple[bytes, PageFit]:
    """Render at the largest font size that fits one A4 page (or the smallest size if nothing fits)."""
    data, pages = b"", 0
    for size in FONT_SIZES:
        data, pages = render_pdf(doc, size)
        if pages <= 1:
            return data, PageFit(pages=1, font_size=size, fits=True)
    return data, PageFit(pages=pages, font_size=FONT_SIZES[-1], fits=False)


# --------------------------------------------------------------------------- DOCX
def render_docx(doc: ResumeDocument, font_size: float = FONT_SIZES[0]) -> bytes:
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Mm, Pt

    font_name = docx_font_name()
    d = docx.Document()
    sec = d.sections[0]
    sec.page_height, sec.page_width = Mm(297), Mm(210)
    sec.left_margin = sec.right_margin = Mm(MARGIN_MM)
    sec.top_margin = sec.bottom_margin = Mm(MARGIN_MM * 0.85)
    text_width = sec.page_width - sec.left_margin - sec.right_margin

    normal = d.styles["Normal"]
    normal.font.name, normal.font.size = font_name, Pt(font_size)
    normal.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), font_name)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.line_spacing = 1.0

    def para(align=None, before: float = 0, after: float = 0):
        p = d.add_paragraph()
        if align is not None:
            p.alignment = align
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(before), Pt(after)
        return p

    def run(p, text: str, bold=False, italic=False, delta: float = 0):
        r = p.add_run(text)
        r.bold, r.italic = bold, italic
        if delta:
            r.font.size = Pt(font_size + delta)
        return r

    def centered(text: str, bold=False, delta: float = 0):
        if _clean(text):
            run(para(WD_ALIGN_PARAGRAPH.CENTER), _clean(text), bold=bold, delta=delta)

    def heading(title: str):
        p = para(before=font_size * 0.8, after=2)
        run(p, title.upper(), bold=True, delta=1.5)
        p_pr = p._p.get_or_add_pPr()
        borders = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        for k, v in {"w:val": "single", "w:sz": "6", "w:space": "1", "w:color": "444444"}.items():
            bottom.set(qn(k), v)
        borders.append(bottom)
        p_pr.append(borders)

    def line(parts, right: str = "", before: float = 0):
        p = para(before=before)
        if right:
            p.paragraph_format.tab_stops.add_tab_stop(text_width, WD_TAB_ALIGNMENT.RIGHT)
        for text, style in parts:
            if text:
                run(p, text, bold="B" in style, italic="I" in style)
        if right:
            run(p, "\t" + right)

    def bullets(items):
        for item in filter(None, map(_clean, items)):
            p = d.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(0)
            run(p, item)

    centered(doc.contact.name or "Resume", bold=True, delta=9)
    centered(doc.contact.headline, delta=1)
    centered(contact_line(doc), delta=-0.5)
    for section in _sections(doc):
        heading(_SECTION_TITLES[section])
        if section == "summary":
            run(para(), _clean(doc.summary))
        elif section == "skills":
            for g in doc.skills:
                items = ", ".join(filter(None, map(_clean, g.items)))
                if items:
                    line([(f"{_clean(g.category)}: " if _clean(g.category) else "", "B"), (items, "")])
        elif section in ("experience", "projects", "education"):
            entries = {"experience": doc.experience, "projects": doc.projects, "education": doc.education}[section]
            parts_of = {"experience": _role_parts, "projects": _project_parts, "education": _edu_parts}[section]
            for i, entry in enumerate(entries):
                parts, right = parts_of(entry)
                line(parts, right, before=font_size * 0.35 if i else 0)
                bullets(getattr(entry, "bullets", []))
        else:
            bullets(getattr(doc, section))

    d.core_properties.title = f"{_clean(doc.contact.name)} Resume".strip()
    d.core_properties.author = _clean(doc.contact.name)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()
