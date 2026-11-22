import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from agents.exceptions import AgentsException
from openai import APIError, OpenAI

from tutorial_generator.agent import generate_tutorial
from tutorial_generator.config import Settings
from tutorial_generator.document import build_tutorial_pdf
from tutorial_generator.research import validate_learning_goal, validate_topic

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
HOST = "127.0.0.1"
PORT = 8001
MAX_REQUEST_BYTES = 8_192


def _filename_for_topic(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.casefold()).strip("-")[:60]
    return f"{slug or 'tutorial'}.pdf"


def make_handler(
    settings: Settings,
    client: OpenAI,
    allowed_origins: set[str],
) -> type[BaseHTTPRequestHandler]:
    class TutorialGeneratorHandler(BaseHTTPRequestHandler):
        def _reply(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _reply_json(self, status: int, payload: dict[str, str]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self._reply(status, "application/json; charset=utf-8", body)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            }
            entry = files.get(path)
            if entry is None:
                self._reply(404, "text/plain; charset=utf-8", b"Not found")
                return

            filename, content_type = entry
            try:
                content = (WEB_DIR / filename).read_bytes()
            except OSError:
                self._reply(500, "text/plain; charset=utf-8", b"Web app unavailable")
                return
            self._reply(200, content_type, content)

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/tutorial":
                self._reply_json(404, {"error": "Not found"})
                return
            if self.headers.get("Origin") not in allowed_origins:
                self._reply_json(403, {"error": "Unexpected request origin"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_REQUEST_BYTES:
                self._reply_json(
                    400,
                    {"error": "Request body is missing or too large."},
                )
                return

            try:
                payload = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._reply_json(400, {"error": "Request body must be valid JSON."})
                return
            if not isinstance(payload, dict):
                self._reply_json(400, {"error": "Request body must be a JSON object."})
                return

            try:
                topic = validate_topic(payload.get("topic"))
                learning_goal = validate_learning_goal(
                    payload.get("learning_goal", "")
                )
            except (TypeError, ValueError) as error:
                self._reply_json(400, {"error": str(error)})
                return

            try:
                tutorial, sources = generate_tutorial(
                    client,
                    settings,
                    topic,
                    learning_goal,
                )
                pdf = build_tutorial_pdf(topic, tutorial, sources)
            except (RuntimeError, ValueError) as error:
                print(f"Tutorial generation failed: {error}", file=sys.stderr)
                self._reply_json(
                    422,
                    {
                        "error": (
                            "The generated tutorial did not meet the quality "
                            f"requirements: {error}"
                        )
                    },
                )
                return
            except APIError as error:
                status = (
                    f" (HTTP {error.status_code})"
                    if error.status_code is not None
                    else ""
                )
                print(f"Tutorial OpenAI API request failed: {error}", file=sys.stderr)
                self._reply_json(
                    502,
                    {
                        "error": (
                            "The OpenAI request failed"
                            f"{status}. Check your API key, model access, "
                            "billing, web-search access, and internet connection."
                        )
                    },
                )
                return
            except (AgentsException, OSError) as error:
                print(f"Tutorial generation failed: {error}", file=sys.stderr)
                self._reply_json(
                    502,
                    {
                        "error": (
                            "The research or tutorial agent could not complete "
                            "the request. Check the server terminal for details."
                        )
                    },
                )
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{_filename_for_topic(topic)}"',
            )
            self.send_header("Content-Length", str(len(pdf)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(pdf)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}", file=sys.stderr)

    return TutorialGeneratorHandler


def main() -> int:
    try:
        settings = Settings.from_env()
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    client = OpenAI(api_key=settings.api_key, max_retries=0)
    origins = {f"http://localhost:{PORT}", f"http://{HOST}:{PORT}"}
    server = ThreadingHTTPServer(
        (HOST, PORT),
        make_handler(settings, client, origins),
    )
    print(f"Tutorial Generator is ready at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTutorial Generator stopped.", file=sys.stderr)
    finally:
        server.server_close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
