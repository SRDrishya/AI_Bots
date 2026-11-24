"""The technology list: built-in entries from the resume's skills plus user-added ones."""
from __future__ import annotations

import re

from app.models import Technology

# Built from the resume's TECHNICAL SKILLS and EXPERIENCE sections, plus the interview-round tracks.
DEFAULT_TECHNOLOGIES: list[Technology] = [
    # Whole-interview banks for the target roles: questions across every skill the role is interviewed on.
    Technology(slug="lead-dotnet-fullstack", name="Lead .NET Full Stack Developer", kind="role",
               group="Target roles", search_terms=".NET full stack developer",
               focus="C#, .NET Core / Web API, Entity Framework, SQL Server, React, Azure DevOps CI/CD, "
                     "design patterns, system design, team leadership, code reviews and release ownership, "
                     "plus the coding problems (DSA) and HR questions asked in these interviews"),
    Technology(slug="dotnet-fullstack-genai", name=".NET Full Stack + Gen AI Developer", kind="role",
               group="Target roles", search_terms=".NET full stack developer generative AI",
               focus="C#, .NET Core / Web API, SQL Server, React, plus Gen AI: LLM APIs, prompt engineering, "
                     "RAG, embeddings and vector search, AI agents, Semantic Kernel / Azure OpenAI integration, "
                     "plus the coding problems (DSA) and HR questions asked in these interviews"),
    Technology(slug="csharp", name="C#", group="Languages", search_terms="C#",
               focus="modern C#: generics, async/await, delegates, records, memory management, collections"),
    Technology(slug="python", name="Python", group="Languages"),
    Technology(slug="javascript", name="JavaScript", group="Languages", focus="ES6+, closures, event loop, promises"),
    Technology(slug="html-css", name="HTML & CSS", group="Languages"),
    Technology(slug="dotnet-core", name=".NET Core / .NET 8-9", group="Backend", search_terms=".NET Core",
               focus="runtime, dependency injection, middleware, configuration, hosting, performance, .NET 8/9"),
    Technology(slug="aspnet-mvc", name="ASP.NET MVC", group="Backend"),
    Technology(slug="aspnet-web-api", name="ASP.NET Core Web API", group="Backend",
               focus="REST design, routing, filters, auth (JWT/OAuth), versioning, validation, error handling"),
    Technology(slug="fastapi", name="FastAPI", group="Backend"),
    Technology(slug="react", name="ReactJS", group="Frontend", search_terms="React",
               focus="hooks, state management, rendering, performance"),
    Technology(slug="entity-framework", name="Entity Framework (EF Core & EF6)", group="Data",
               search_terms="Entity Framework",
               focus="DbContext, migrations, tracking, loading strategies, performance, EF Core vs EF6"),
    Technology(slug="linq", name="LINQ", group="Data", search_terms="LINQ C#"),
    Technology(slug="ado-net", name="ADO.NET", group="Data"),
    Technology(slug="sql-server", name="SQL Server", group="Data",
               focus="T-SQL, indexing, query tuning, stored procedures, transactions & isolation levels"),
    Technology(slug="postgresql", name="PostgreSQL", group="Data"),
    Technology(slug="azure-devops", name="Azure DevOps & CI/CD", group="Tools & DevOps",
               search_terms="Azure DevOps CI/CD"),
    Technology(slug="git-github", name="Git & GitHub", group="Tools & DevOps", search_terms="Git"),
    Technology(slug="unit-testing", name="Unit Testing (xUnit & Moq)", group="Tools & DevOps",
               search_terms="unit testing xUnit Moq C#"),
    Technology(slug="system-design", name="System Design & Microservices", kind="design",
               group="Interview rounds", search_terms="system design microservices",
               focus="scalability, microservices, messaging, caching, API gateways, integration patterns"),
    Technology(slug="design-patterns", name="Design Patterns & SOLID", kind="design", group="Interview rounds",
               search_terms="design patterns SOLID principles C#"),
    Technology(slug="behavioral", name="Behavioral & Leadership", kind="behavioral", group="Interview rounds",
               search_terms="behavioral interview software engineer",
               focus="leadership, code reviews, conflict, failure, ownership, delivery pressure"),
    Technology(slug="ai-llm-rag", name="AI / LLM & RAG", kind="ai", group="Interview rounds",
               search_terms="LLM RAG generative AI engineer",
               focus="LLM basics, prompting, RAG pipelines, embeddings, vector search, AI agents"),
]
# AmbitionBox pages where candidates list the questions they were asked, per company and job role
# (ambitionbox.com/interviews/<company>-interview-questions/<role>). Questions about other skills on
# these pages are skipped during extraction, so the .NET pages feed every .NET-related bank.
DOTNET_ROLES = ["dot-net-developer", "senior-dot-net-developer"]
AMBITIONBOX_ROLES = {
    "lead-dotnet-fullstack": [*DOTNET_ROLES, "full-stack-developer"],
    "dotnet-fullstack-genai": [*DOTNET_ROLES, "full-stack-developer"],
    "python": ["python-developer"], "fastapi": ["python-developer"],
    "react": ["full-stack-developer"], "javascript": ["full-stack-developer"],
    "html-css": ["full-stack-developer"], "postgresql": ["full-stack-developer"],
    "system-design": ["senior-software-engineer", *DOTNET_ROLES],
    "behavioral": ["senior-software-engineer", "dot-net-developer"],
    "ai-llm-rag": ["ai-engineer"],
}
for _tech in DEFAULT_TECHNOLOGIES:
    _tech.builtin = True
    _tech.ambitionbox_roles = AMBITIONBOX_ROLES.get(_tech.slug, DOTNET_ROLES)


def slugify(name: str) -> str:
    text = name.lower().replace("#", "sharp").replace("+", "plus").replace(".", "dot")
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:60] or "tech"


def merge(custom: list[Technology]) -> list[Technology]:
    """Built-in entries first, then custom ones (a custom slug never overrides a built-in one)."""
    builtin_slugs = {t.slug for t in DEFAULT_TECHNOLOGIES}
    return [*DEFAULT_TECHNOLOGIES, *(t for t in custom if t.slug not in builtin_slugs)]


def find(technologies: list[Technology], slug: str) -> Technology | None:
    return next((t for t in technologies if t.slug == slug), None)


def same_name(a: str, b: str) -> bool:
    return slugify(a) == slugify(b)
