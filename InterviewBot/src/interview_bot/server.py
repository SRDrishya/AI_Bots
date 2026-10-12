import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

from openai import APIError, OpenAI
from openai.types.live.media_session_config_param import MediaSessionConfigParam

from interview_bot.config import Settings
from interview_bot.topic_notes import (
    DOCUMENTS_DIR,
    prepare_topic_notes,
    search_documents,
)

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
NOTES_DIR = Path(__file__).resolve().parents[2] / "topic_notes"
HOST = "127.0.0.1"
PORT = 8000
MAX_REQUEST_BYTES = 65_536
MAX_TOOL_REQUEST_BYTES = 8_192

TOPIC_NOTE_TOOLS = [
    {
        "type": "function",
        "name": "search_documents",
        "description": (
            "Search the local documents folder and saved topic notes for "
            "passages relevant to the selected topic and current question. "
            "Call this before answering. Judge whether the returned passages "
            "adequately support a useful answer and question. If there are "
            "no passages or they are insufficient, call prepare_topic_notes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "query": {"type": "string"},
            },
            "required": ["topic", "query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "prepare_topic_notes",
        "description": (
            "Research a topic with web search and GPT-6 Luna, write short "
            "sourced notes, save a PDF in the documents folder and a Markdown "
            "copy in topic_notes, and return the notes. Call only when the "
            "available local documents are missing or insufficient."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "candidate_context": {
                    "type": "string",
                },
            },
            "required": ["topic", "candidate_context"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def build_session_config(settings: Settings) -> MediaSessionConfigParam:
    return {
        "model": settings.model,
        "instructions": settings.interviewer_instructions,
        "audio": {"output": {"voice": settings.voice}},
        "delegation": {
            "type": "responses",
            "responses": {
                "model": settings.reasoning_model,
                "instructions": settings.reasoning_instructions,
                "reasoning": {"effort": "medium"},
                "max_output_tokens": 256,
                "tools": TOPIC_NOTE_TOOLS,
                "parallel_tool_calls": False,
            },
        },
    }


def execute_topic_tool(
    name: object,
    arguments: object,
    client: OpenAI,
    settings: Settings,
    notes_dir: Path = NOTES_DIR,
    documents_dir: Path = DOCUMENTS_DIR,
) -> dict[str, object]:
    if not isinstance(arguments, dict):
        raise TypeError("Tool arguments must be a JSON object.")

    if name == "search_documents":
        return search_documents(
            topic=cast(str, arguments.get("topic")),
            query=cast(str, arguments.get("query")),
            notes_dir=notes_dir,
            documents_dir=documents_dir,
        )
    if name == "prepare_topic_notes":
        return prepare_topic_notes(
            client=client,
            settings=settings,
            topic=cast(str, arguments.get("topic")),
            candidate_context=cast(str, arguments.get("candidate_context")),
            notes_dir=notes_dir,
            documents_dir=documents_dir,
        )
    raise ValueError("Unsupported topic-notes tool.")


def make_handler(
    settings: Settings,
    client: OpenAI,
    allowed_origins: set[str],
    notes_dir: Path = NOTES_DIR,
    documents_dir: Path = DOCUMENTS_DIR,
) -> type[BaseHTTPRequestHandler]:
    class InterviewBotHandler(BaseHTTPRequestHandler):
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
            files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            }
            entry = files.get(self.path)
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
            if self.path not in {"/api/session", "/api/tool"}:
                self._reply_json(404, {"error": "Not found"})
                return
            if self.headers.get("Origin") not in allowed_origins:
                self._reply_json(403, {"error": "Unexpected request origin"})
                return

            max_length = (
                MAX_REQUEST_BYTES
                if self.path == "/api/session"
                else MAX_TOOL_REQUEST_BYTES
            )
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= max_length:
                self._reply_json(400, {"error": "Request body is missing or too large"})
                return

            try:
                payload = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._reply_json(400, {"error": "Request body must be valid JSON"})
                return

            if self.path == "/api/tool":
                try:
                    result = execute_topic_tool(
                        name=payload.get("name") if isinstance(payload, dict) else None,
                        arguments=(
                            payload.get("arguments")
                            if isinstance(payload, dict)
                            else None
                        ),
                        client=client,
                        settings=settings,
                        notes_dir=notes_dir,
                        documents_dir=documents_dir,
                    )
                except (TypeError, ValueError) as error:
                    self._reply_json(400, {"error": str(error)})
                    return
                except APIError as error:
                    status = error.status_code
                    if status is None or not 400 <= status < 600:
                        status = 502
                    print(
                        f"Topic-notes API request failed (HTTP {status}).",
                        file=sys.stderr,
                    )
                    self._reply_json(
                        status,
                        {"error": "Could not research this topic right now."},
                    )
                    return
                except (OSError, RuntimeError) as error:
                    print(f"Topic-notes operation failed: {error}", file=sys.stderr)
                    self._reply_json(
                        502,
                        {"error": "Could not prepare or read the topic notes."},
                    )
                    return
                self._reply_json(
                    200,
                    {"result": json.dumps(result, ensure_ascii=False)},
                )
                return

            sdp = payload.get("sdp") if isinstance(payload, dict) else None
            if not isinstance(sdp, str) or not sdp.strip():
                self._reply_json(400, {"error": "An SDP offer is required"})
                return

            try:
                result = client.live.create(
                    session=build_session_config(settings),
                    transport={"type": "webrtc", "sdp": sdp},
                )
            except APIError as error:
                status = error.status_code
                if status is None or not 400 <= status < 600:
                    status = 502
                print(
                    f"GPT-Live session creation failed (HTTP {status}).",
                    file=sys.stderr,
                )
                self._reply_json(status, {"error": "Could not start the interview"})
                return

            body = result.model_dump_json().encode("utf-8")
            self._reply(201, "application/json; charset=utf-8", body)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}", file=sys.stderr)

    return InterviewBotHandler


def main() -> int:
    try:
        settings = Settings.from_env()
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    client = OpenAI(api_key=settings.api_key, max_retries=0)
    origins = {f"http://localhost:{PORT}", f"http://{HOST}:{PORT}"}
    server = ThreadingHTTPServer(
        (HOST, PORT), make_handler(settings, client, origins)
    )
    print(f"InterviewBot is ready at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nInterviewBot stopped.", file=sys.stderr)
    finally:
        server.server_close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
