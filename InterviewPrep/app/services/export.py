"""Turn the questions selected on the web page into a PDF, Word (DOCX) or HTML study guide."""
from __future__ import annotations

import html
import io
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.models import LEVELS, Bank, Question

LEVEL_LABEL = {"beginner": "Beginner", "intermediate": "Intermediate", "advanced": "Advanced"}


@dataclass
class ExportOptions:
    title: str
    subtitle: str
    include_sources: bool = True
    include_angle: bool = True


def ordered(bank: Bank, questions: list[Question]) -> list[tuple[str, list[Question]]]:
    """Questions grouped by category (bank order), most likely to be asked first inside each group."""
    order = {c: i for i, c in enumerate(bank.categories)}
    groups: dict[str, list[Question]] = {}
    for q in questions:
        groups.setdefault(q.category, []).append(q)
    return [(cat, sorted(groups[cat], key=lambda q: (-q.likelihood, LEVELS.index(q.level))))
            for cat in sorted(groups, key=lambda c: order.get(c, len(order)))]


def evidence(q: Question) -> str:
    """'Reported by candidates at Infosys, TCS · found on 3 sites' (empty when there is nothing to say)."""
    if q.origin == "ai":
        return "AI-generated question (Claude Code)"
    parts = []
    if q.reported:
        parts.append("Reported by candidates" + (f" at {', '.join(q.companies)}" if q.companies else ""))
    elif q.companies:
        parts.append(f"Asked at {', '.join(q.companies)}")
    sites = len({s.url.split('/')[2] for s in q.sources if '//' in s.url})
    if sites > 1:
        parts.append(f"found on {sites} sites")
    return " · ".join(parts)


def strip_inline(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)


def blocks(markdown: str) -> list[tuple[str, str]]:
    """Very small Markdown reader: ('code'|'bullet'|'para', text)."""
    out, para, code, in_code = [], [], [], False

    def flush():
        if para:
            out.append(("para", " ".join(para)))
            para.clear()

    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            if in_code:
                out.append(("code", "\n".join(code)))
                code = []
            else:
                flush()
            in_code = not in_code
            continue
        if in_code:
            code.append(line)
            continue
        bullet = re.match(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)", line)
        if bullet or not line.strip() or line.lstrip().startswith("#"):
            flush()
            if bullet:
                out.append(("bullet", bullet.group(1)))
            elif line.strip():
                out.append(("para", line.lstrip("# ").strip()))
            continue
        para.append(line.strip())
    flush()
    if code:
        out.append(("code", "\n".join(code)))
    return out


# ---------- HTML ----------

def _inline_html(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"\[([^\]]+)\]\((https?:[^)\s]+)\)", r'<a href="\2">\1</a>', text)


def _md_html(markdown: str) -> str:
    parts, bullets = [], []
    for kind, text in blocks(markdown):
        if kind == "bullet":
            bullets.append(f"<li>{_inline_html(text)}</li>")
            continue
        if bullets:
            parts.append(f"<ul>{''.join(bullets)}</ul>")
            bullets = []
        parts.append(f"<pre><code>{html.escape(text)}</code></pre>" if kind == "code"
                     else f"<p>{_inline_html(text)}</p>")
    if bullets:
        parts.append(f"<ul>{''.join(bullets)}</ul>")
    return "".join(parts)


HTML_STYLE = """
body{font:15px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;max-width:860px;margin:40px auto;
padding:0 20px;color:#1d2330;background:#fff}
h1{margin-bottom:0}.sub{color:#667085;margin-top:4px}h2{color:#2f5fd0;border-bottom:1px solid #e4e7ec;
padding-bottom:4px;margin-top:40px}h3{margin:28px 0 4px}.lvl{font-size:12px;color:#667085}
.sum{font-weight:600}pre{background:#f2f4f7;border:1px solid #e4e7ec;padding:10px 12px;border-radius:8px;
overflow-x:auto;font-size:13px}code{font-family:Consolas,Menlo,monospace;background:#f2f4f7;padding:1px 4px;
border-radius:4px}pre code{padding:0}.angle{background:#e3f4ec;border-left:3px solid #137a4b;padding:8px 12px;
border-radius:6px}.src{font-size:12px;color:#667085}.toc a{text-decoration:none}
@media print{body{margin:0}h3{page-break-after:avoid}}
"""


def to_html(bank: Bank, questions: list[Question], opts: ExportOptions) -> str:
    groups = ordered(bank, questions)
    toc = "".join(f'<li><a href="#c{i}">{html.escape(c)}</a> ({len(qs)})</li>' for i, (c, qs) in enumerate(groups))
    body, number = [], 0
    for i, (category, items) in enumerate(groups):
        body.append(f'<h2 id="c{i}">{html.escape(category)}</h2>')
        for q in items:
            number += 1
            ev = f" · {html.escape(evidence(q))}" if evidence(q) else ""
            body.append(f"<h3>{number}. {html.escape(q.question)}</h3><div class='lvl'>{LEVEL_LABEL[q.level]}{ev}</div>")
            a = q.answer
            if a is None:
                body.append("<p><i>No answer yet.</i></p>")
            else:
                if a.summary:
                    body.append(f"<p class='sum'>{_inline_html(a.summary)}</p>")
                body.append(_md_html(a.explanation))
                if a.code_example:
                    body.append(f"<pre><code>{html.escape(a.code_example)}</code></pre>")
                for title, items_ in (("Key points", a.key_points), ("Common pitfalls", a.pitfalls),
                                      ("Likely follow-up questions", a.follow_ups)):
                    if items_:
                        lis = "".join(f"<li>{_inline_html(x)}</li>" for x in items_)
                        body.append(f"<p><b>{title}</b></p><ul>{lis}</ul>")
                if opts.include_angle and a.your_angle:
                    body.append(f"<p class='angle'><b>Your angle:</b> {_inline_html(a.your_angle)}</p>")
            if opts.include_sources and q.sources:
                links = " · ".join(f'<a href="{html.escape(s.url)}">{html.escape(s.title)}</a>' for s in q.sources)
                body.append(f"<p class='src'>Source: {links}</p>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(opts.title)}</title>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'><style>{HTML_STYLE}</style></head>"
            f"<body><h1>{html.escape(opts.title)}</h1><p class='sub'>{html.escape(opts.subtitle)}</p>"
            f"<ul class='toc'>{toc}</ul>{''.join(body)}</body></html>")


# ---------- DOCX ----------

def to_docx(bank: Bank, questions: list[Question], opts: ExportOptions) -> bytes:
    from docx import Document
    from docx.enum.text import WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)
    muted, green = RGBColor(0x66, 0x70, 0x85), RGBColor(0x13, 0x7A, 0x4B)

    def rich(paragraph, text: str, bold: bool = False, italic: bool = False, color=None) -> None:
        """Adds text, turning **bold** and `code` spans into formatted runs."""
        for part in re.split(r"(\*\*.+?\*\*|`[^`]+`)", text):
            if not part:
                continue
            run = paragraph.add_run(part.strip("*`") if part[:1] in "*`" else part)
            run.bold = bold or part.startswith("**")
            run.italic = italic
            if part.startswith("`"):
                run.font.name = "Consolas"
            if color is not None:
                run.font.color.rgb = color

    def code(text: str) -> None:
        p = doc.add_paragraph()
        shading = OxmlElement("w:shd")
        shading.set(qn("w:val"), "clear")
        shading.set(qn("w:fill"), "F2F4F7")
        p._p.get_or_add_pPr().append(shading)
        for i, line in enumerate(text.split("\n")):
            run = p.add_run(line)
            run.font.name, run.font.size = "Consolas", Pt(9)
            if i < text.count("\n"):
                run.add_break(WD_BREAK.LINE)

    doc.add_heading(opts.title, 0)
    rich(doc.add_paragraph(), opts.subtitle, color=muted)
    number = 0
    for category, items in ordered(bank, questions):
        doc.add_heading(category, 1)
        for q in items:
            number += 1
            doc.add_heading(f"{number}. {q.question}", 3)
            ev = f" · {evidence(q)}" if evidence(q) else ""
            rich(doc.add_paragraph(), LEVEL_LABEL[q.level] + ev, italic=True, color=muted)
            a = q.answer
            if a is None:
                rich(doc.add_paragraph(), "No answer yet.", italic=True)
            else:
                if a.summary:
                    rich(doc.add_paragraph(), a.summary, bold=True)
                for kind, text in blocks(a.explanation):
                    if kind == "code":
                        code(text)
                    else:
                        rich(doc.add_paragraph(style="List Bullet" if kind == "bullet" else None), text)
                if a.code_example:
                    code(a.code_example)
                for title, items_ in (("Key points", a.key_points), ("Common pitfalls", a.pitfalls),
                                      ("Likely follow-up questions", a.follow_ups)):
                    if items_:
                        rich(doc.add_paragraph(), title, bold=True)
                        for item in items_:
                            rich(doc.add_paragraph(style="List Bullet"), item)
                if opts.include_angle and a.your_angle:
                    rich(doc.add_paragraph(), f"Your angle: {a.your_angle}", italic=True, color=green)
            if opts.include_sources and q.sources:
                rich(doc.add_paragraph(), "Source: " + " · ".join(f"{s.title} ({s.url})" for s in q.sources),
                     color=muted)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


# ---------- PDF ----------

FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/Library/Fonts")]
TEXT_FONTS = [("Arial", "arial.ttf", "arialbd.ttf", "ariali.ttf"),
              ("DejaVu", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf")]
CODE_FONTS = [("Consolas", "consola.ttf"), ("DejaVuMono", "DejaVuSansMono.ttf")]


def _find_font(name: str) -> Path | None:
    return next((d / name for d in FONT_DIRS if (d / name).exists()), None)


def to_pdf(bank: Bank, questions: list[Question], opts: ExportOptions) -> bytes:
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)

    text_font, code_font, unicode_ok = "Helvetica", "Courier", False
    for family, regular, bold, italic in TEXT_FONTS:
        paths = [_find_font(f) for f in (regular, bold, italic)]
        if all(paths):
            for style, path in zip(("", "B", "I"), paths):
                pdf.add_font(family, style, str(path))
            text_font, unicode_ok = family, True
            break
    for family, regular in CODE_FONTS:
        path = _find_font(regular)
        if path and unicode_ok:
            pdf.add_font(family, "", str(path))
            code_font = family
            break
    bullet = "•  " if unicode_ok else "-  "

    def clean(text: str) -> str:
        return text if unicode_ok else text.encode("latin-1", "replace").decode("latin-1")

    def write(text: str, size: float = 10, style: str = "", indent: float = 0, color=(30, 35, 48), link=None):
        pdf.set_font(text_font, style, size)
        pdf.set_text_color(*color)
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(pdf.epw - indent, size * 0.5, clean(text), new_x="LMARGIN", new_y="NEXT", link=link)

    def code_block(text: str) -> None:
        pdf.set_font(code_font, "", 8.5)
        pdf.set_fill_color(242, 244, 247)
        pdf.set_text_color(30, 35, 48)
        pdf.multi_cell(pdf.epw, 4.2, clean(text.replace("\t", "    ")), fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    def bullets(title: str, items: list[str]) -> None:
        if items:
            write(title, 10, "B")
            for item in items:
                write(bullet + strip_inline(item), indent=4)
            pdf.ln(1.5)

    pdf.add_page()
    write(opts.title, 20, "B")
    write(opts.subtitle, 11, color=(102, 112, 133))
    pdf.ln(4)
    number = 0
    for category, items in ordered(bank, questions):
        pdf.ln(2)
        write(category, 15, "B", color=(47, 95, 208))
        pdf.ln(1)
        for q in items:
            number += 1
            if pdf.get_y() > pdf.h - 50:
                pdf.add_page()
            write(f"{number}. {q.question}", 11.5, "B")
            ev = f" · {evidence(q)}" if evidence(q) else ""
            write(LEVEL_LABEL[q.level] + ev, 8.5, "I", color=(102, 112, 133))
            pdf.ln(1)
            a = q.answer
            if a is None:
                write("No answer yet.", 10, "I")
            else:
                if a.summary:
                    write(strip_inline(a.summary), 10, "B")
                    pdf.ln(1)
                for kind, text in blocks(a.explanation):
                    if kind == "code":
                        code_block(text)
                    elif kind == "bullet":
                        write(bullet + strip_inline(text), indent=4)
                    else:
                        write(strip_inline(text))
                        pdf.ln(1)
                if a.code_example:
                    code_block(a.code_example)
                bullets("Key points", a.key_points)
                bullets("Common pitfalls", a.pitfalls)
                bullets("Likely follow-up questions", a.follow_ups)
                if opts.include_angle and a.your_angle:
                    write(f"Your angle: {strip_inline(a.your_angle)}", 9.5, "I", color=(19, 122, 75))
            if opts.include_sources:
                for s in q.sources:
                    write(f"Source: {s.title}", 8, color=(47, 95, 208), link=s.url)
            pdf.ln(4)
    return bytes(pdf.output())


def subtitle(count: int, experience_label: str) -> str:
    return f"{count} interview questions · answers for {experience_label} · {date.today():%d %b %Y}"
