import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from docx import Document
from openai import OpenAI

from interview_bot.config import Settings
from interview_bot.prompts import DEFAULT_INTERVIEWER_INSTRUCTIONS
from interview_bot.server import (
    TOPIC_NOTE_TOOLS,
    build_session_config,
    execute_topic_tool,
    make_handler,
)
from interview_bot.topic_notes import (
    prepare_topic_notes,
    search_documents,
    topic_document_path,
    topic_note_path,
    validate_topic,
)


class SessionConfigTests(unittest.TestCase):
    def start_test_server(
        self,
        client: OpenAI,
        notes_dir: Path | None = None,
        documents_dir: Path | None = None,
    ) -> ThreadingHTTPServer:
        resolved_notes_dir = notes_dir if notes_dir is not None else Path("topic_notes")
        resolved_documents_dir = (
            documents_dir
            if documents_dir is not None
            else resolved_notes_dir.parent / "documents"
        )
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(
                Settings(api_key="test-key"),
                client,
                {"http://localhost"},
                notes_dir=resolved_notes_dir,
                documents_dir=resolved_documents_dir,
            ),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        return server

    def test_web_assets_are_served(self) -> None:
        server = self.start_test_server(cast(OpenAI, Mock()))

        with urlopen(f"http://127.0.0.1:{server.server_port}/") as response:
            page = response.read().decode()
        with urlopen(f"http://127.0.0.1:{server.server_port}/app.js") as response:
            script = response.read().decode()

        self.assertIn("Start interview", page)
        self.assertIn("Pause interview", page)
        self.assertIn("RTCPeerConnection", script)
        self.assertIn("session.input_audio.mute", script)
        self.assertIn("session.input_audio.unmute", script)
        self.assertIn("response.output_item.done", script)
        self.assertIn("/api/tool", script)
        self.assertIn("what topic they want to discuss and their preferred level", script)

    def test_session_uses_live_model_prompt_and_voice(self) -> None:
        settings = Settings(
            api_key="test-key",
            model="gpt-live-1",
            voice="marin",
            interviewer_instructions="Interview me.",
        )

        session = build_session_config(settings)

        self.assertEqual(session["model"], "gpt-live-1")
        self.assertEqual(session["instructions"], "Interview me.")
        self.assertEqual(session["audio"], {"output": {"voice": "marin"}})
        self.assertEqual(
            session["delegation"],
            {
                "type": "responses",
                "responses": {
                    "model": "gpt-6-luna",
                    "instructions": settings.reasoning_instructions,
                    "reasoning": {"effort": "medium"},
                    "max_output_tokens": 256,
                    "tools": TOPIC_NOTE_TOOLS,
                    "parallel_tool_calls": False,
                },
            },
        )
        self.assertNotIn("input", session)

    def test_prompts_assign_questions_to_responses_backend(self) -> None:
        settings = Settings(api_key="test-key")

        session = build_session_config(settings)

        self.assertEqual(
            session["instructions"], DEFAULT_INTERVIEWER_INSTRUCTIONS
        )
        self.assertIn(
            "Delegate the opening and every candidate turn",
            session["instructions"],
        )
        self.assertIn(
            "ask exactly one relevant question grounded in retrieved evidence",
            session["delegation"]["responses"]["instructions"],
        )
        self.assertIn("search_documents", session["delegation"]["responses"]["instructions"])
        self.assertIn("prepare_topic_notes", session["delegation"]["responses"]["instructions"])
        self.assertIn(
            "Do not use web search when local documents",
            session["delegation"]["responses"]["instructions"],
        )

    def test_session_route_exchanges_sdp_and_returns_live_answer(self) -> None:
        settings = Settings(api_key="test-key")
        live_response = {
            "session": {"id": "live_test"},
            "transport": {"type": "webrtc", "sdp": "answer-sdp"},
        }
        create_session = Mock(
            return_value=SimpleNamespace(
                model_dump_json=lambda: json.dumps(live_response)
            )
        )
        client = cast(
            OpenAI, SimpleNamespace(live=SimpleNamespace(create=create_session))
        )
        server = self.start_test_server(client)

        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/session",
            data=json.dumps({"sdp": "offer-sdp"}).encode(),
            headers={"Content-Type": "application/json", "Origin": "http://localhost"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.loads(response.read())

        self.assertEqual(response.status, 201)
        self.assertEqual(payload["transport"]["sdp"], "answer-sdp")
        create_session.assert_called_once_with(
            session=build_session_config(settings),
            transport={"type": "webrtc", "sdp": "offer-sdp"},
        )

    def test_session_route_rejects_unexpected_origin(self) -> None:
        server = self.start_test_server(cast(OpenAI, Mock()))

        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/session",
            data=b'{"sdp":"offer-sdp"}',
            headers={"Content-Type": "application/json", "Origin": "https://attacker"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(request)

        self.assertEqual(raised.exception.code, 403)

    def test_tool_route_returns_not_found_for_missing_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server = self.start_test_server(
                cast(OpenAI, Mock()),
                root / "topic_notes",
                root / "documents",
            )
            request = Request(
                f"http://127.0.0.1:{server.server_port}/api/tool",
                data=json.dumps(
                    {
                        "name": "search_documents",
                        "arguments": {"topic": "RAG", "query": "chunking"},
                    }
                ).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Origin": "http://localhost",
                },
                method="POST",
            )
            with urlopen(request) as response:
                payload = json.loads(response.read())

        self.assertEqual(response.status, 200)
        self.assertFalse(json.loads(payload["result"])["found"])

    def test_tool_route_rejects_invalid_argument_types(self) -> None:
        server = self.start_test_server(cast(OpenAI, Mock()))
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/tool",
            data=json.dumps(
                {
                    "name": "search_documents",
                    "arguments": {"topic": None, "query": "chunking"},
                }
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "Origin": "http://localhost",
            },
            method="POST",
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(request)

        self.assertEqual(raised.exception.code, 400)


class TopicNotesTests(unittest.TestCase):
    def test_topic_path_is_safe_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = topic_note_path("../../RAG & Retrieval", Path(temp_dir))
            self.assertEqual(path.parent, Path(temp_dir))
            self.assertTrue(path.name.startswith("rag-retrieval-"))

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            validate_topic("  ")

    def test_search_reports_missing_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = search_documents(
                "RAG", "chunking", root / "topic_notes", root / "documents"
            )

        self.assertFalse(result["found"])
        self.assertEqual(result["documents"], [])

    def test_prepared_notes_are_saved_with_real_web_citations(self) -> None:
        citations = [
            SimpleNamespace(
                type="url_citation",
                url="https://example.org/rag",
                title="RAG overview",
            )
        ]
        api_response = SimpleNamespace(
            output_text="- Retrieval searches external documents.\n"
            "- Generation uses retrieved evidence.",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            annotations=citations,
                        )
                    ],
                )
            ],
        )
        client = cast(
            OpenAI,
            SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=api_response))),
        )
        settings = Settings(api_key="test-key")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = prepare_topic_notes(
                client,
                settings,
                "RAG",
                "Beginner",
                root / "topic_notes",
                root / "documents",
            )
            saved_file = topic_note_path("RAG", root / "topic_notes")
            content = saved_file.read_text(encoding="utf-8")
            pdf_path = topic_document_path("RAG", root / "documents")
            pdf_content = pdf_path.read_bytes()
            retrieved = search_documents(
                "RAG",
                "retrieval",
                root / "topic_notes",
                root / "documents",
            )

        self.assertTrue(result["prepared"])
        self.assertIn("[RAG overview](https://example.org/rag)", content)
        self.assertEqual(
            result["sources"],
            [{"title": "RAG overview", "url": "https://example.org/rag"}],
        )
        self.assertEqual(result["document"], pdf_path.name)
        self.assertTrue(pdf_content.startswith(b"%PDF"))
        self.assertTrue(retrieved["found"])
        self.assertTrue(
            any(item["document"] == pdf_path.name for item in retrieved["documents"])
        )
        client.responses.create.assert_called_once()
        self.assertEqual(
            client.responses.create.call_args.kwargs["tools"],
            [{"type": "web_search"}],
        )

    def test_prepare_refuses_to_save_notes_without_citations(self) -> None:
        api_response = SimpleNamespace(
            output_text="- An uncited claim.",
            output=[],
        )
        client = cast(
            OpenAI,
            SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=api_response))),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "no verifiable source citations"):
                prepare_topic_notes(
                    client,
                    Settings(api_key="test-key"),
                    "RAG",
                    "",
                    Path(temp_dir) / "topic_notes",
                    Path(temp_dir) / "documents",
                )
            self.assertFalse((Path(temp_dir) / "documents").exists())

    def test_search_retrieves_matching_markdown_passages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            notes_dir = root / "topic_notes"
            note_file = topic_note_path("RAG", notes_dir)
            note_file.parent.mkdir(parents=True, exist_ok=True)
            note_file.write_text(
                "# RAG\n\n## Short notes\n\n"
                "- Chunking splits long documents into searchable passages.\n"
                "- Retrieval selects passages relevant to the query.\n"
                "- Generation uses retrieved context to answer.\n\n"
                "## Sources\n\n- [Guide](https://example.org/rag)\n",
                encoding="utf-8",
            )

            result = search_documents(
                "RAG",
                "How does chunking work?",
                notes_dir,
                root / "documents",
            )

        self.assertTrue(result["found"])
        self.assertTrue(
            any("Chunking splits" in item["passage"] for item in result["documents"])
        )

    def test_search_reads_docx_and_plain_text_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_dir = Path(temp_dir) / "documents"
            documents_dir.mkdir()
            (documents_dir / "rag.txt").write_text(
                "RAG retrieval selects useful passages from source documents.",
                encoding="utf-8",
            )
            docx = Document()
            docx.add_paragraph("Chunking splits documents into searchable pieces.")
            docx.save(documents_dir / "chunking.docx")

            result = search_documents(
                "RAG",
                "How does chunking work?",
                Path(temp_dir) / "topic_notes",
                documents_dir,
            )

        passages = [item["passage"] for item in result["documents"]]
        self.assertTrue(any("Chunking splits" in passage for passage in passages))
        self.assertTrue(any("retrieval selects" in passage for passage in passages))

    def test_search_tool_is_routed_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_topic_tool(
                "search_documents",
                {"topic": "RAG", "query": "retrieval"},
                cast(OpenAI, Mock()),
                Settings(api_key="test-key"),
                Path(temp_dir),
                Path(temp_dir) / "documents",
            )
        self.assertFalse(result["found"])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            execute_topic_tool(
                "unknown_tool",
                {},
                cast(OpenAI, Mock()),
                Settings(api_key="test-key"),
                Path(temp_dir),
            )


if __name__ == "__main__":
    unittest.main()
