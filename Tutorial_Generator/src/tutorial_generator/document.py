import re
from collections.abc import Iterable
from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Flowable,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)

from tutorial_generator.research import Source, is_http_url

MAX_DIAGRAM_NODES = 8
_INLINE_TOKEN = re.compile(
    r"(\*\*[^*]+\*\*|(?<!\*)\*[^*]+\*|`[^`]+`|\[[^\]]+\]\(https?://[^)\s]+\))"
)


def _inline_markup(text: str) -> str:
    parts: list[str] = []
    position = 0
    for match in _INLINE_TOKEN.finditer(text):
        parts.append(escape(text[position : match.start()]))
        token = match.group()
        if token.startswith("**"):
            parts.append(f"<b>{escape(token[2:-2])}</b>")
        elif token.startswith("*"):
            parts.append(f"<i>{escape(token[1:-1])}</i>")
        elif token.startswith("`"):
            parts.append(f'<font name="Courier">{escape(token[1:-1])}</font>')
        else:
            link = re.fullmatch(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", token)
            if link is None:
                parts.append(escape(token))
            else:
                label, url = link.groups()
                parts.append(
                    f'<link href="{escape(url, quote=True)}">{escape(label)}</link>'
                )
        position = match.end()
    parts.append(escape(text[position:]))
    return "".join(parts)


def parse_diagram(source: str) -> tuple[str, ...] | None:
    nodes = tuple(part.strip() for part in source.split("->"))
    if not 2 <= len(nodes) <= MAX_DIAGRAM_NODES:
        return None
    if any(not node or len(node) > 80 for node in nodes):
        return None
    return nodes


class FlowDiagram(Flowable):
    def __init__(self, nodes: Iterable[str]) -> None:
        super().__init__()
        self.nodes = tuple(nodes)
        self.paragraphs: list[Paragraph] = []
        self.box_heights: list[float] = []
        self.box_width = 0.0
        self.gap = 22.0

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        self.width = available_width
        self.box_width = min(400.0, available_width - 24.0)
        style = ParagraphStyle(
            "DiagramNode",
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            alignment=1,
            textColor=colors.HexColor("#17324d"),
        )
        self.paragraphs = [
            Paragraph(_inline_markup(node), style) for node in self.nodes
        ]
        self.box_heights = []
        for paragraph in self.paragraphs:
            _, text_height = paragraph.wrap(self.box_width - 20.0, available_height)
            self.box_heights.append(max(34.0, text_height + 14.0))
        self.height = sum(self.box_heights) + self.gap * (len(self.nodes) - 1)
        return self.width, self.height

    def draw(self) -> None:
        x = (self.width - self.box_width) / 2
        current_y = self.height
        for index, (paragraph, box_height) in enumerate(
            zip(self.paragraphs, self.box_heights)
        ):
            current_y -= box_height
            self.canv.setFillColor(colors.HexColor("#eef6ff"))
            self.canv.setStrokeColor(colors.HexColor("#3975a8"))
            self.canv.roundRect(x, current_y, self.box_width, box_height, 7, fill=1)
            _, text_height = paragraph.wrap(self.box_width - 20.0, box_height)
            paragraph.drawOn(
                self.canv,
                x + 10.0,
                current_y + (box_height - text_height) / 2,
            )
            if index < len(self.nodes) - 1:
                arrow_x = self.width / 2
                arrow_top = current_y - 2
                arrow_bottom = current_y - self.gap + 4
                self.canv.setStrokeColor(colors.HexColor("#3975a8"))
                self.canv.setFillColor(colors.HexColor("#3975a8"))
                self.canv.line(arrow_x, arrow_top, arrow_x, arrow_bottom + 4)
                self.canv.line(arrow_x, arrow_bottom + 4, arrow_x - 4, arrow_bottom + 9)
                self.canv.line(arrow_x, arrow_bottom + 4, arrow_x + 4, arrow_bottom + 9)
                current_y -= self.gap


def _markdown_flowables(
    markdown: str,
    styles: dict[str, ParagraphStyle],
) -> list[Flowable]:
    story: list[Flowable] = []
    paragraph_lines: list[str] = []
    code_lines: list[str] = []
    code_kind: str | None = None

    def flush_paragraph() -> None:
        if paragraph_lines:
            story.append(
                Paragraph(
                    _inline_markup(" ".join(paragraph_lines)),
                    styles["BodyText"],
                )
            )
            story.append(Spacer(1, 5))
            paragraph_lines.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if code_kind is not None:
            if line.strip().startswith("```"):
                code = "\n".join(code_lines).strip("\n")
                nodes = parse_diagram(code) if code_kind == "diagram" else None
                if nodes is not None:
                    story.append(FlowDiagram(nodes))
                else:
                    story.append(Preformatted(code, styles["TutorialCode"]))
                story.append(Spacer(1, 8))
                code_kind = None
                code_lines.clear()
            else:
                code_lines.append(line)
            continue

        if line.strip().startswith("```"):
            flush_paragraph()
            code_kind = line.strip()[3:].strip().casefold()
            continue
        if not line.strip():
            flush_paragraph()
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            story.append(
                Paragraph(
                    _inline_markup(heading.group(2)),
                    styles[f"Heading{level}"],
                )
            )
            story.append(Spacer(1, 4))
            continue

        bullet = re.match(r"^\s*[-*]\s+(.+)$", line)
        numbered = re.match(r"^\s*(\d+)[.)]\s+(.+)$", line)
        if bullet or numbered:
            flush_paragraph()
            if bullet:
                text = bullet.group(1)
                bullet_text = "-"
            else:
                assert numbered is not None
                text = numbered.group(2)
                bullet_text = f"{numbered.group(1)}."
            story.append(
                Paragraph(
                    _inline_markup(text),
                    styles["BodyText"],
                    bulletText=bullet_text,
                )
            )
            story.append(Spacer(1, 3))
            continue
        if re.fullmatch(r"\s*[-*_]{3,}\s*", line):
            flush_paragraph()
            story.append(Spacer(1, 6))
            continue

        paragraph_lines.append(line.strip())

    if code_kind is not None:
        flush_paragraph()
        story.append(Preformatted("\n".join(code_lines), styles["TutorialCode"]))
    else:
        flush_paragraph()
    return story


def build_tutorial_pdf(
    topic: str,
    markdown: str,
    sources: tuple[Source, ...],
) -> bytes:
    for source in sources:
        if not is_http_url(source.url):
            raise ValueError("Tutorial sources must use valid HTTP or HTTPS links.")

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "TutorialCode",
            parent=styles["BodyText"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            leftIndent=8,
            rightIndent=8,
            spaceBefore=4,
            spaceAfter=6,
        )
    )
    output = BytesIO()
    story: list[Flowable] = [
        Paragraph(_inline_markup(topic), styles["Title"]),
        Spacer(1, 12),
    ]
    story.extend(_markdown_flowables(markdown, styles))
    story.extend(
        [
            Spacer(1, 12),
            Paragraph("Sources", styles["Heading2"]),
            Spacer(1, 6),
        ]
    )
    for index, source in enumerate(sources, start=1):
        title = escape(source.title)
        url = escape(source.url, quote=True)
        story.append(
            Paragraph(
                f'[{index}] <link href="{url}">{title}</link>',
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 4))

    SimpleDocTemplate(
        output,
        pagesize=letter,
        title=topic,
        author="Tutorial Generator",
    ).build(story)
    return output.getvalue()
