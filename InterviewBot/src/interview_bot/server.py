import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agents.exceptions import AgentsException
from openai import APIError, OpenAI
from openai.types.live.media_session_config_param import MediaSessionConfigParam

from interview_bot.agent import create_topic_rag_agent, run_topic_rag_agent
from interview_bot.config import Settings
from interview_bot.prompts import DEFAULT_AGENT_DISPATCH_INSTRUCTIONS
from interview_bot.topic_notes import DOCUMENTS_DIR

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
NOTES_DIR = Path(__file__).resolve().parents[2] / "topic_notes"
HOST = "127.0.0.1"
PORT = 8000
MAX_REQUEST_BYTES = 65_536
MAX_AGENT_REQUEST_BYTES = 32_768

INTERVIEW_AGENT_TOOLS = [
    {
        "type": "function",
        "name": "run_topic_rag_agent",
        "description": (
            "Ask the interview RAG agent to produce the next spoken response. "
            "Include the full available conversation, the candidate's current "
            "answer, and the chosen topic and level. The agent searches local "
            "documents first and decides whether web research is necessary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "conversation_context": {"type": "string"},
            },
            "required": ["conversation_context"],
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
                "instructions": DEFAULT_AGENT_DISPATCH_INSTRUCTIONS,
                "reasoning": {"effort": "medium"},
                "max_output_tokens": 256,
                "tools": INTERVIEW_AGENT_TOOLS,
                "parallel_tool_calls": False,
            },
        },
    }


def make_handler(
    settings: Settings,
    client: OpenAI,
    allowed_origins: set[str],
    notes_dir: Path = NOTES_DIR,
    documents_dir: Path = DOCUMENTS_DIR,
) -> type[BaseHTTPRequestHandler]:
    agent = create_topic_rag_agent(
        settings=settings,
        client=client,
        notes_dir=notes_dir,
        documents_dir=documents_dir,
    )

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
            if self.path not in {"/api/session", "/api/agent"}:
                self._reply_json(404, {"error": "Not found"})
                return
            if self.headers.get("Origin") not in allowed_origins:
                self._reply_json(403, {"error": "Unexpected request origin"})
                return

            max_length = (
                MAX_REQUEST_BYTES
                if self.path == "/api/session"
                else MAX_AGENT_REQUEST_BYTES
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

            if self.path == "/api/agent":
                try:
                    context = (
                        payload.get("conversation_context")
                        if isinstance(payload, dict)
                        else None
                    )
                    result = run_topic_rag_agent(agent, context)
                except (TypeError, ValueError) as error:
                    self._reply_json(400, {"error": str(error)})
                    return
                except (AgentsException, APIError, OSError, RuntimeError) as error:
                    print(
                        f"Interview agent request failed: {error}",
                        file=sys.stderr,
                    )
                    self._reply_json(
                        502,
                        {"error": "The interview agent could not complete this turn."},
                    )
                    return
                self._reply_json(
                    200,
                    {"result": result},
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
