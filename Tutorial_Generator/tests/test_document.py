from io import BytesIO

import pytest
from pypdf import PdfReader

from tutorial_generator.document import build_tutorial_pdf, parse_diagram
from tutorial_generator.research import Source


def test_parse_diagram_validates_node_count_and_labels() -> None:
    assert parse_diagram("Question -> Search -> Learn") == (
        "Question",
        "Search",
        "Learn",
    )
    assert parse_diagram("Only one") is None
    assert parse_diagram(" -> Empty") is None
    assert parse_diagram(" -> ".join(["Step"] * 9)) is None


def test_build_tutorial_pdf_includes_diagram_and_linked_sources() -> None:
    pdf = build_tutorial_pdf(
        "How a web request works",
        (
            "## Overview\n"
            "A browser sends a request and receives a response. [1]\n\n"
            "## Visual overview\n"
            "```diagram\nBrowser -> Server -> Response\n```\n\n"
            "## Example or pseudocode\n"
            "```python\nprint('Hello')\n```"
        ),
        (
            Source(
                title="HTTP overview",
                url="https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview",
            ),
        ),
    )

    reader = PdfReader(BytesIO(pdf))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert pdf.startswith(b"%PDF")
    assert reader.metadata.title == "How a web request works"
    assert "How a web request works" in text
    assert "Browser" in text
    assert "Server" in text
    assert "Response" in text
    assert "HTTP overview" in text
    annotations = reader.pages[0].get("/Annots", [])
    link_urls = {
        annotation.get_object()["/A"]["/URI"]
        for annotation in annotations
        if annotation.get_object().get("/A", {}).get("/URI")
    }
    assert "https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview" in link_urls


def test_build_tutorial_pdf_rejects_invalid_source_urls() -> None:
    with pytest.raises(ValueError):
        build_tutorial_pdf(
            "A topic",
            "## Overview\nA short overview.",
            (Source(title="Unsafe", url="javascript:alert(1)"),),
        )
