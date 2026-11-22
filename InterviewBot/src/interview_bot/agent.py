import json
from pathlib import Path

from agents import Agent, Runner
from agents.decorators import tool
from openai import OpenAI

from interview_bot.config import Settings
from interview_bot.topic_notes import (
    DOCUMENTS_DIR,
    prepare_topic_notes,
    search_documents,
)

MAX_AGENT_CONTEXT_LENGTH = 24_000
MAX_AGENT_TURNS = 8


def create_topic_rag_agent(
    settings: Settings,
    client: OpenAI,
    notes_dir: Path,
    documents_dir: Path = DOCUMENTS_DIR,
) -> Agent:
    @tool
    def search_local_documents(topic: str, query: str) -> str:
        """Retrieve ranked, relevant passages from the available local files.

        Search local PDF, DOCX, Markdown, and TXT files plus prior generated
        topic notes. Use the returned passages and document names to decide
        whether local information adequately supports the requested answer.
        """
        result = search_documents(
            topic=topic,
            query=query,
            notes_dir=notes_dir,
            documents_dir=documents_dir,
        )
        return json.dumps(result, ensure_ascii=False)

    @tool
    def prepare_sourced_topic_notes(topic: str, candidate_context: str) -> str:
        """Research and save a cited topic brief when local files are insufficient.

        Use web search only when relevant local passages are unavailable or do
        not adequately support the candidate's current topic or question.
        """
        result = prepare_topic_notes(
            client=client,
            settings=settings,
            topic=topic,
            candidate_context=candidate_context,
            notes_dir=notes_dir,
            documents_dir=documents_dir,
        )
        return json.dumps(result, ensure_ascii=False)

    return Agent(
        name="Interview Topic RAG Agent",
        instructions=settings.reasoning_instructions,
        model=settings.reasoning_model,
        tools=[search_local_documents, prepare_sourced_topic_notes],
    )


def run_topic_rag_agent(agent: Agent, conversation_context: str) -> str:
    if not isinstance(conversation_context, str):
        raise TypeError("Conversation context must be text.")
    context = conversation_context.strip()
    if not context:
        raise ValueError("Conversation context cannot be empty.")
    if len(context) > MAX_AGENT_CONTEXT_LENGTH:
        context = context[-MAX_AGENT_CONTEXT_LENGTH:]

    result = Runner.run_sync(agent, context, max_turns=MAX_AGENT_TURNS)
    if not isinstance(result.final_output, str) or not result.final_output.strip():
        raise RuntimeError("The interview agent did not return a spoken response.")
    return result.final_output.strip()
