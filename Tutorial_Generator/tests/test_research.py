from types import SimpleNamespace

import pytest

from tutorial_generator.config import Settings
from tutorial_generator.research import (
    Source,
    extract_web_citations,
    is_http_url,
    research_topic,
    validate_learning_goal,
    validate_topic,
)


def test_validate_topic_normalizes_whitespace() -> None:
    assert validate_topic("  neural\n networks  ") == "neural networks"


def test_validate_topic_rejects_empty_and_long_values() -> None:
    with pytest.raises(ValueError, match="Enter a topic"):
        validate_topic(" \n ")
    with pytest.raises(ValueError, match="160 characters"):
        validate_topic("x" * 161)


def test_validate_learning_goal_rejects_non_text() -> None:
    with pytest.raises(TypeError, match="Learning goal"):
        validate_learning_goal(None)


def test_extract_web_citations_returns_unique_http_sources() -> None:
    citations = [
        SimpleNamespace(
            type="url_citation",
            title="Python Tutorial",
            url="https://docs.python.org/tutorial/",
        ),
        SimpleNamespace(
            type="url_citation",
            title="Same source",
            url="https://docs.python.org/tutorial/",
        ),
        SimpleNamespace(
            type="url_citation",
            title="Unsafe",
            url="javascript:alert(1)",
        ),
    ]
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", annotations=citations),
                ],
            ),
        ],
    )

    assert extract_web_citations(response) == (
        Source(
            title="Python Tutorial",
            url="https://docs.python.org/tutorial/",
        ),
    )


def test_http_url_requires_a_host_and_rejects_whitespace() -> None:
    assert is_http_url("https://example.com/tutorial")
    assert not is_http_url("https://")
    assert not is_http_url("https://example.com/bad link")
    assert not is_http_url("javascript:alert(1)")


def test_research_requests_detailed_source_notes() -> None:
    citation = SimpleNamespace(
        type="url_citation",
        title="RAG documentation",
        url="https://example.com/rag",
    )
    response = SimpleNamespace(
        output_text="Detailed source-supported research.",
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(
                        type="output_text",
                        annotations=[citation],
                    ),
                ],
            ),
        ],
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: setattr(client, "request", kwargs) or response
        )
    )

    result = research_topic(client, Settings(api_key="test"), "RAG")

    assert result.notes == "Detailed source-supported research."
    assert client.request["max_output_tokens"] == 5_000
    assert "substantial, accurate research notes" in client.request["instructions"]
    assert "not a brief or a handful of bullets" in client.request["input"]
