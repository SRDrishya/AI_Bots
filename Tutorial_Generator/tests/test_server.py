import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openai import OpenAI
from pypdf import PdfReader

import tutorial_generator.server as server_module
from tutorial_generator.config import Settings
from tutorial_generator.research import Source
from tutorial_generator.server import make_handler


@contextmanager
def running_server(settings: Settings, client: OpenAI) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    host, port = server.server_address
    origin = f"http://{host}:{port}"
    server.RequestHandlerClass = make_handler(
        settings,
        client,
        {origin, f"http://localhost:{port}"},
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield origin
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_tutorial_endpoint_returns_a_downloadable_pdf(monkeypatch) -> None:
    client = OpenAI(api_key="test")
    settings = Settings(api_key="test")
    source = Source(title="HTTP overview", url="https://example.com/http")
    monkeypatch.setattr(
        server_module,
        "generate_tutorial",
        lambda *_args: (
            (
                "## Overview\nA browser sends a request.\n"
                "## Visual overview\n```diagram\nBrowser -> Server\n```"
            ),
            (source,),
        ),
    )

    with running_server(settings, client) as base_url:
        request = Request(
            f"{base_url}/api/tutorial",
            data=json.dumps({"topic": "Web Request"}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": base_url,
            },
            method="POST",
        )
        with urlopen(request) as response:
            pdf = response.read()
            assert response.status == 200
            assert response.headers["Content-Type"] == "application/pdf"
            assert (
                response.headers["Content-Disposition"]
                == 'attachment; filename="web-request.pdf"'
            )

    client.close()
    assert pdf.startswith(b"%PDF")
    document_text = "\n".join(
        page.extract_text() or "" for page in PdfReader(BytesIO(pdf)).pages
    )
    assert "Web Request" in document_text
    assert "HTTP overview" in document_text


def test_tutorial_endpoint_rejects_an_untrusted_origin() -> None:
    client = OpenAI(api_key="test")
    settings = Settings(api_key="test")
    with running_server(settings, client) as base_url:
        request = Request(
            f"{base_url}/api/tutorial",
            data=b'{"topic":"A topic"}',
            headers={
                "Content-Type": "application/json",
                "Origin": "https://untrusted.example",
            },
            method="POST",
        )
        try:
            urlopen(request)
        except HTTPError as error:
            assert error.code == 403
            assert json.loads(error.read()) == {"error": "Unexpected request origin"}
        else:
            raise AssertionError("An untrusted request origin should be rejected.")

    client.close()


def test_tutorial_endpoint_shows_quality_validation_error(monkeypatch) -> None:
    client = OpenAI(api_key="test")
    settings = Settings(api_key="test")
    monkeypatch.setattr(
        server_module,
        "generate_tutorial",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("The tutorial is too short (210 words).")
        ),
    )

    with running_server(settings, client) as base_url:
        request = Request(
            f"{base_url}/api/tutorial",
            data=json.dumps({"topic": "RAG"}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": base_url,
            },
            method="POST",
        )
        try:
            urlopen(request)
        except HTTPError as error:
            assert error.code == 422
            assert json.loads(error.read()) == {
                "error": (
                    "The generated tutorial did not meet the quality "
                    "requirements: The tutorial is too short (210 words)."
                )
            }
        else:
            raise AssertionError("A failed tutorial should return an error.")

    client.close()
