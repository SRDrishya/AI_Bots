"""Prompt templates. Kept in one place so they are easy to tune.

The AI never writes interview questions. It (1) searches the web for pages, (2) copies questions that
are on those pages, (3) judges duplicates, and (4) writes answers for the selected experience level.
"""
from __future__ import annotations

from app.config import Experience
from app.models import Technology


def candidate(experience: Experience) -> str:
    return f"a software engineer with {experience.label.split(' (')[0]} of experience"


def _topic(tech: Technology) -> str:
    return f"{tech.name} (focus: {tech.focus})" if tech.focus else tech.name


def _resume_block(resume: str) -> str:
    return f"<resume>\n{resume.strip()[:12000]}\n</resume>" if resume.strip() else "<resume>(not provided)</resume>"


def plan(tech: Technology, count: int, experience: Experience) -> str:
    """Category names only — used to group the questions found on the web."""
    if tech.kind == "behavioral":
        scope = ("Cover areas like leadership, conflict, failure, ownership, stakeholder management, "
                 "mentoring, delivery pressure and decision making.")
    elif tech.kind == "role":
        scope = ("This is a whole job interview, so make each category one skill area the role is interviewed "
                 "on (e.g. 'C# & OOP', '.NET Core & Web API', 'SQL Server', 'React', 'Leadership & Process'), "
                 "covering every skill in the focus list. Always include the categories 'Coding Problems' "
                 "(DSA / logic programs asked in the interview) and 'HR & Behavioral'.")
    else:
        scope = ("Together they must cover the subject end to end: fundamentals, internals, common real-world "
                 "scenarios, performance, security, testing / debugging and recent version features where relevant.")
    return (
        f"Plan exactly {count} categories for grouping interview questions on {_topic(tech)} asked of "
        f"{candidate(experience)}. {scope} Category names are 1-4 words; give each a one-line focus. "
        "No overlap between categories."
    )


def discover(query: str) -> str:
    return (
        f"Use web search to find web pages with interview questions for: \"{query}\".\n"
        "Most valuable are pages where real candidates report the questions they were actually asked: "
        "AmbitionBox interview questions and company interview pages, Glassdoor interviews, GeeksforGeeks "
        "interview experiences, Reddit threads, LeetCode Discuss. Interview-preparation articles and GitHub "
        "question lists are useful too. Run a few different searches.\n"
        "Then list every useful page you found, one per line, as: - <page title> | <url>\n"
        "Do not write any interview questions yourself."
    )


def extract(tech: Technology, experience: Experience, categories: list[str], text: str, url: str) -> str:
    cats = ", ".join(categories)
    scope = (f"any skill this role is interviewed on ({tech.focus})" if tech.kind == "role" else _topic(tech))
    return (
        f"Below is text downloaded from {url}. Copy out the interview questions about {scope} that appear "
        "in it.\n\nRules:\n"
        "- Copy each question as written on the page. You may only remove numbering, fix obvious typos, and "
        "turn an interview report line (e.g. 'they asked me about middleware order') into a short question "
        "that keeps the page's words.\n"
        "- If a question is cut off ('...'), complete only the last word or two when the ending is obvious "
        "(e.g. 'a web ap...' -> 'a web application?'); otherwise skip it.\n"
        "- Skip vague topic labels that are not real questions (e.g. 'Basic Dotnet question', '.net "
        "fundamentals', 'Azure question').\n"
        "- NEVER write a question that is not on the page. If the page has no questions, return an empty list.\n"
        f"- Skip questions that are not about {tech.name}, and skip answers, ads and navigation text.\n"
        f"- category: the best fit from this list: {cats}\n"
        "- level: beginner, intermediate or advanced\n"
        f"- fits_experience: true if this is typical for interviews of {candidate(experience)} "
        f"(expected depth: {experience.expectation}); false only if it is a basic question usually asked "
        "of freshers / juniors at this level, or far too specialised.\n"
        "- reported_by_candidate: true if the page presents it as asked in a real interview (a candidate's "
        "interview report or experience, 'asked in <company>', an interview review); false for prep lists "
        "and articles.\n"
        "- companies: the companies the page says asked it, e.g. ['Infosys']; empty list if none is named.\n\n"
        f"<page_text>\n{text}\n</page_text>"
    )


def same_questions(pairs: list[tuple[int, str, str]]) -> str:
    lines = "\n".join(f"{n}. A: {a}\n   B: {b}" for n, a, b in pairs)
    return (
        "For each numbered pair of interview questions, decide whether they ask the SAME thing, so a "
        "candidate would give the same answer to both. Different subjects (e.g. IEnumerable vs IQueryable "
        "compared with IEnumerable vs ICollection) are NOT the same. Return one verdict per pair.\n\n" + lines
    )


def answers(tech: Technology, experience: Experience, numbered: list[tuple[int, str]], resume: str) -> str:
    questions = "\n".join(f"{n}. {q}" for n, q in numbered)
    who = candidate(experience)
    if tech.kind == "behavioral":
        style = (
            "These are behavioral questions. In `explanation`, write a model answer in STAR format "
            "(**Situation**, **Task**, **Action**, **Result**) built from the candidate's real experience in the "
            "resume, at the scope expected for their experience level, followed by a short note on what the "
            "interviewer is looking for. Never invent numbers or facts; where a metric would help, use a "
            "placeholder like [X%] or [N days] for the candidate to fill. Leave code_language and code_example empty."
        )
    else:
        style = (
            "For a behavioral or leadership question, answer in STAR format from the resume (placeholders like "
            "[X%] instead of invented numbers). For an HR question (notice period, salary, why leave, tell me "
            "about yourself), give a short, honest framing the candidate can adapt, built on the resume. For a "
            "coding problem, explain the approach and its time / space complexity, and put a clean C# solution "
            "in code_example. For a technical question: "
            "in `explanation`, explain how it works and why, trade-offs, when to use / avoid it, and a short "
            "real-world example. Keep the explanation general: do not mention the candidate's employers, clients "
            "or projects there (that belongs in your_angle). Add a code_example whenever code makes the answer "
            "clearer; it must be correct and idiomatic for current versions. Otherwise leave code_language and "
            "code_example empty."
        )
    return (
        f"You are a senior interviewer and expert in {_topic(tech)}. Write model interview answers for {who}.\n"
        f"Pitch every answer at that level. Interviewers expect: {experience.expectation}. Go beyond a "
        "textbook definition wherever the level calls for it.\n\n"
        f"{style}\n\n"
        "For each numbered question return:\n"
        "- number: the question's number\n"
        "- summary: the 1-2 sentence direct answer the candidate should open with\n"
        "- explanation: the detailed answer in Markdown (about 150-350 words; short paragraphs, bullet lists "
        "and **bold** allowed; no headings; put code in code_example, not here)\n"
        "- code_language and code_example (plain code, no ``` fences, formatted over multiple lines with "
        "normal indentation exactly as it would appear in an editor), or empty strings\n"
        "- key_points: 3-6 crisp points worth saying out loud\n"
        "- pitfalls: 1-4 common mistakes or misconceptions\n"
        "- follow_ups: 2-4 follow-up questions an interviewer would ask at this experience level\n"
        "- your_angle: 1-2 sentences of advice TO the candidate on linking the answer to their resume. Name "
        "the real resume fact, and present the technique only as a question for them to check, because the "
        "resume does not tell you which techniques they used. "
        "Good: 'Your benefit-calculation APIs on SQL Server are a natural example: if you used ROW_NUMBER() "
        "there (e.g. to pick the latest policy version), describe that case.' "
        "Bad: 'Mention that you used ROW_NUMBER() to retrieve the latest policy version.' (claims something "
        "the resume doesn't say). Only state a technique as used if the resume itself names it. Empty string "
        "if nothing in the resume relates\n\n"
        f"Candidate resume:\n{_resume_block(resume)}\n\nQuestions:\n{questions}"
    )


def evaluate(tech_name: str, experience: Experience, question: str, reference: str, answer: str,
             resume: str) -> str:
    return (
        f"You are a fair but demanding interviewer hiring {candidate(experience)}. Score the candidate's "
        f"answer to this {tech_name} interview question against what that level requires "
        f"({experience.expectation}).\n\n<question>{question}</question>\n\n"
        f"<reference_answer>\n{reference or '(none — use your own expertise)'}\n</reference_answer>\n\n"
        f"<candidate_answer>\n{answer}\n</candidate_answer>\n\n"
        "Return: score (integer 0-10, where 7+ would pass at this experience level), verdict (one sentence), "
        "strengths (what was good), gaps (missing or wrong points, most important first), and improved_answer "
        "(a tightened version of THEIR answer, in their voice, that would score 9-10; Markdown allowed). "
        "Judge correctness and depth, not wording.\n\n"
        f"Candidate background:\n{_resume_block(resume)}"
    )
