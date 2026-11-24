"""Practice mode: score the user's own answer against the model answer."""
from __future__ import annotations

from app.config import Experience
from app.models import Evaluation, EvaluationDraft, Question
from app.services import ai, prompts


def reference_text(question: Question) -> str:
    answer = question.answer
    if answer is None:
        return ""
    parts = [answer.summary, answer.explanation]
    if answer.key_points:
        parts.append("Key points:\n" + "\n".join(f"- {p}" for p in answer.key_points))
    if answer.code_example:
        parts.append(f"Example code ({answer.code_language}):\n{answer.code_example}")
    return "\n\n".join(p for p in parts if p)


async def evaluate(tech_name: str, experience: Experience, question: Question, user_answer: str,
                   resume: str) -> Evaluation:
    draft = await ai.generate_json(
        prompt=prompts.evaluate(tech_name, experience, question.question, reference_text(question), user_answer, resume),
        schema=EvaluationDraft,
        temperature=0.1,
        fast=True,
    )
    return Evaluation(
        score=max(0, min(10, draft.score)),
        verdict=draft.verdict.strip(),
        strengths=[s for s in draft.strengths if s.strip()],
        gaps=[g for g in draft.gaps if g.strip()],
        improved_answer=draft.improved_answer.strip(),
    )
