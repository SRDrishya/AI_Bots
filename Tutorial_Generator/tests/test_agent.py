import json
from types import SimpleNamespace

import pytest

import tutorial_generator.agent as agent_module
from tutorial_generator.agent import (
    _research_context,
    generate_tutorial,
    validate_tutorial_content,
)
from tutorial_generator.config import Settings
from tutorial_generator.research import ResearchResult, Source

SOURCES = (
    Source(title="Python tutorial", url="https://docs.python.org/tutorial/"),
)


def valid_lesson() -> str:
    explanation = (
        "A clear example helps beginners understand each concept. " * 30
    )
    sections = (
        "In Simple Words",
        "Imagine It This Way",
        "How It Works",
        "Example",
        "Visual Overview",
        "When It Is Useful",
        "One Limitation",
        "Quick Recap",
    )
    content = "".join(
        f"## {section}\n{explanation}[1]\n\n" for section in sections
    )
    content += "```diagram\nInput -> Process -> Output\n```"
    return content


def test_validate_tutorial_requires_diagram_and_citation() -> None:
    validate_tutorial_content(valid_lesson(), SOURCES)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (valid_lesson().replace("```diagram", "```text"), "valid flow diagram"),
        (valid_lesson().replace("[1]", "[2]"), "invalid source citation"),
        (
            valid_lesson().replace("Input -> Process -> Output", "Only one node"),
            "valid flow diagram",
        ),
    ],
    ids=("missing-diagram", "invalid-citation", "invalid-diagram"),
)
def test_validate_tutorial_rejects_missing_or_invalid_required_content(
    content: str,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_tutorial_content(content, SOURCES)


def test_validate_tutorial_rejects_short_content() -> None:
    short_lesson = valid_lesson().replace(
        "A clear example helps beginners understand each concept. " * 30,
        "Brief explanation. ",
    )
    with pytest.raises(RuntimeError, match="at least 350 words"):
        validate_tutorial_content(short_lesson, SOURCES)


def test_validate_tutorial_requires_the_complete_lesson_outline() -> None:
    content = valid_lesson().replace("## How It Works", "## Process")

    with pytest.raises(RuntimeError, match="how it works"):
        validate_tutorial_content(content, SOURCES)


def test_validate_tutorial_rejects_interview_practice_section() -> None:
    content = valid_lesson().replace(
        "## Quick Recap",
        "## Interview Practice\n"
        "1. **Question:** What is the concept?\n"
        "**Answer:** A detailed answer.\n\n"
        "## Quick Recap",
    )

    with pytest.raises(RuntimeError, match="separate feature"):
        validate_tutorial_content(content, SOURCES)


def test_validate_tutorial_rejects_bullet_only_explanations() -> None:
    content = valid_lesson().replace(
        "A clear example helps beginners understand each concept. ",
        "- A clear example helps beginners understand each concept.\n",
    )

    with pytest.raises(RuntimeError, match="enough prose"):
        validate_tutorial_content(content, SOURCES)


def test_research_context_numbers_sources_for_inline_citations() -> None:
    result = ResearchResult(
        notes="Research notes about the topic.",
        sources=(
            Source(title="First source", url="https://example.com/one"),
            Source(title="Second source", url="https://example.com/two"),
        ),
    )

    output = json.loads(_research_context(result))

    assert output["sources"] == [
        {
            "number": 1,
            "title": "First source",
            "url": "https://example.com/one",
        },
        {
            "number": 2,
            "title": "Second source",
            "url": "https://example.com/two",
        },
    ]


def test_generate_tutorial_researches_before_calling_writer(monkeypatch) -> None:
    research_result = ResearchResult(
        notes="Verified research notes.",
        sources=SOURCES,
    )
    calls: list[tuple[object, str, int]] = []

    monkeypatch.setattr(
        agent_module,
        "research_topic",
        lambda **_kwargs: research_result,
    )

    def fake_run_sync(agent: object, task: str, max_turns: int) -> SimpleNamespace:
        calls.append((agent, task, max_turns))
        return SimpleNamespace(final_output=valid_lesson())

    monkeypatch.setattr(
        agent_module.Runner,
        "run_sync",
        staticmethod(fake_run_sync),
    )

    tutorial, sources = generate_tutorial(
        client=object(),
        settings=Settings(api_key="test"),
        topic="RAG",
        learning_goal="Explain the basic idea",
    )

    assert tutorial == valid_lesson()
    assert sources == SOURCES
    assert len(calls) == 1
    task = json.loads(calls[0][1])
    assert task["verified_web_research"]["research"] == "Verified research notes."
    assert task["verified_web_research"]["sources"][0]["number"] == 1
