from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from openai import OpenAI

from tutorial_generator.config import Settings
from tutorial_generator.prompts import RESEARCH_INSTRUCTIONS

MAX_TOPIC_LENGTH = 160
MAX_GOAL_LENGTH = 500
MAX_RESEARCH_LENGTH = 24_000
MAX_SOURCES = 8


@dataclass(frozen=True, slots=True)
class Source:
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class ResearchResult:
    notes: str
    sources: tuple[Source, ...]


def is_http_url(url: object) -> bool:
    if not isinstance(url, str) or any(character.isspace() for character in url):
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_topic(topic: object) -> str:
    if not isinstance(topic, str):
        raise TypeError("Topic must be text.")
    normalized = " ".join(topic.split())
    if not normalized:
        raise ValueError("Enter a topic to get started.")
    if len(normalized) > MAX_TOPIC_LENGTH:
        raise ValueError(f"Topic must be {MAX_TOPIC_LENGTH} characters or fewer.")
    return normalized


def validate_learning_goal(goal: object) -> str:
    if not isinstance(goal, str):
        raise TypeError("Learning goal must be text.")
    normalized = " ".join(goal.split())
    if len(normalized) > MAX_GOAL_LENGTH:
        raise ValueError(
            f"Learning goal must be {MAX_GOAL_LENGTH} characters or fewer."
        )
    return normalized


def extract_web_citations(response: Any) -> tuple[Source, ...]:
    sources: dict[str, Source] = {}
    for output in getattr(response, "output", []):
        if getattr(output, "type", None) != "message":
            continue
        for content in getattr(output, "content", []):
            if getattr(content, "type", None) != "output_text":
                continue
            for annotation in getattr(content, "annotations", []):
                if getattr(annotation, "type", None) != "url_citation":
                    continue
                url = getattr(annotation, "url", "")
                if not is_http_url(url):
                    continue
                title = getattr(annotation, "title", "") or url
                sources.setdefault(url, Source(title=str(title), url=url))
                if len(sources) == MAX_SOURCES:
                    return tuple(sources.values())
    return tuple(sources.values())


def research_topic(
    client: OpenAI,
    settings: Settings,
    topic: str,
    learning_goal: str = "",
) -> ResearchResult:
    normalized_topic = validate_topic(topic)
    normalized_goal = validate_learning_goal(learning_goal)
    goal = normalized_goal or "Build a clear beginner-level overview."

    response = client.responses.create(
        model=settings.model,
        instructions=RESEARCH_INSTRUCTIONS,
        input=(
            f"Topic: {normalized_topic}\n"
            f"Learner's goal: {goal}\n"
            "Search relevant tutorials and reliable source material. "
            "Return detailed explanatory research notes, not a brief or a "
            "handful of bullets. Explain the concept, its parts and process, "
            "a worked example, practical uses, and limitations. Support facts "
            "with numbered citations."
        ),
        tools=[{"type": "web_search"}],
        max_output_tokens=5_000,
    )
    notes = response.output_text.strip()
    if not notes:
        raise RuntimeError("Web research returned no tutorial notes.")
    if len(notes) > MAX_RESEARCH_LENGTH:
        raise RuntimeError("Web research returned more text than the agent can use.")

    sources = extract_web_citations(response)
    if not sources:
        raise RuntimeError(
            "Web research returned no verifiable source links; no tutorial was created."
        )
    return ResearchResult(notes=notes, sources=sources)
