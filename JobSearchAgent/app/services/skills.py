"""Skill taxonomy shared by resume parsing and job scoring.

Each canonical skill maps to the aliases that should count as a mention of it.
Matching is case-insensitive and boundary-aware, so "Java" does not match
"JavaScript" and "C" is never matched on its own (too ambiguous).
"""
from __future__ import annotations

import re
from functools import lru_cache

SKILL_ALIASES: dict[str, list[str]] = {
    # Languages
    "Python": ["python"],
    "Java": ["java"],
    "JavaScript": ["javascript", "js", "es6"],
    "TypeScript": ["typescript"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp"],
    "Go": ["golang"],
    "Rust": ["rust"],
    "Kotlin": ["kotlin"],
    "Swift": ["swift"],
    "Scala": ["scala"],
    "Ruby": ["ruby"],
    "PHP": ["php"],
    "SQL": ["sql"],
    "Bash": ["bash", "shell scripting"],
    "Dart": ["dart"],
    # Web / frameworks
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular", "angularjs"],
    "Vue.js": ["vue", "vue.js", "vuejs"],
    "Next.js": ["next.js", "nextjs"],
    "Node.js": ["node.js", "nodejs", "node"],
    "Express": ["express.js", "expressjs"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Spring Boot": ["spring boot", "springboot", "spring framework", "spring mvc"],
    ".NET": [".net", "dotnet", "asp.net"],
    "Ruby on Rails": ["rails", "ruby on rails"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3", "tailwind", "sass"],
    "GraphQL": ["graphql"],
    "REST APIs": ["restful", "rest api", "rest apis"],
    "Flutter": ["flutter"],
    "React Native": ["react native"],
    # Data / ML / AI
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision", "opencv"],
    "LLMs": ["llm", "llms", "large language models", "genai", "generative ai", "gen ai"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "LangChain": ["langchain", "langgraph"],
    "PyTorch": ["pytorch"],
    "TensorFlow": ["tensorflow", "keras"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Spark": ["spark", "pyspark", "apache spark"],
    "Hadoop": ["hadoop"],
    "Airflow": ["airflow"],
    "Kafka": ["kafka"],
    "dbt": ["dbt"],
    "Snowflake": ["snowflake"],
    "Databricks": ["databricks"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["excel"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Statistics": ["statistics", "statistical"],
    "ETL": ["etl", "elt"],
    # Databases
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "Elasticsearch": ["elasticsearch", "opensearch"],
    "Oracle": ["oracle"],
    "SQL Server": ["sql server", "mssql"],
    "DynamoDB": ["dynamodb"],
    # Cloud / DevOps
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Terraform": ["terraform"],
    "Ansible": ["ansible"],
    "Jenkins": ["jenkins"],
    "CI/CD": ["ci/cd", "cicd", "github actions", "gitlab ci"],
    "Linux": ["linux", "unix"],
    "Git": ["git", "github", "gitlab", "bitbucket"],
    "Microservices": ["microservices", "microservice"],
    "System Design": ["system design", "distributed systems"],
    # Testing / QA
    "Selenium": ["selenium"],
    "Cypress": ["cypress"],
    "Playwright": ["playwright"],
    "Jest": ["jest"],
    "Pytest": ["pytest"],
    "Test Automation": ["test automation", "automation testing"],
    # Enterprise / other
    "Salesforce": ["salesforce"],
    "SAP": ["sap"],
    "ServiceNow": ["servicenow"],
    "Agile": ["agile", "scrum", "kanban"],
    "Jira": ["jira"],
    "Figma": ["figma"],
    "Product Management": ["product management", "product roadmap"],
    "Project Management": ["project management", "pmp"],
}

# Role titles we try to detect in a resume (order = rough priority).
KNOWN_ROLES: list[str] = [
    "Machine Learning Engineer",
    "AI Engineer",
    "Data Scientist",
    "Data Engineer",
    "Data Analyst",
    "Business Analyst",
    "Full Stack Developer",
    "Frontend Developer",
    "Backend Developer",
    "Mobile Developer",
    "Android Developer",
    "iOS Developer",
    "DevOps Engineer",
    "Site Reliability Engineer",
    "Cloud Engineer",
    "QA Engineer",
    "Test Automation Engineer",
    "Software Engineer",
    "Software Developer",
    "Solutions Architect",
    "Engineering Manager",
    "Product Manager",
    "Project Manager",
    "UI/UX Designer",
    "Salesforce Developer",
    "SAP Consultant",
]

# Extra spellings that should map onto a KNOWN_ROLES entry.
ROLE_ALIASES: dict[str, list[str]] = {
    "Machine Learning Engineer": ["ml engineer"],
    "Full Stack Developer": ["full stack engineer", "fullstack developer", "full-stack developer", "full-stack engineer"],
    "Frontend Developer": ["front end developer", "front-end developer", "frontend engineer", "front-end engineer", "ui developer"],
    "Backend Developer": ["back end developer", "back-end developer", "backend engineer", "back-end engineer"],
    "Site Reliability Engineer": ["sre"],
    "QA Engineer": ["quality assurance engineer", "qa analyst", "software tester", "test engineer"],
    "Software Engineer": ["sde", "software development engineer", "programmer analyst"],
    "UI/UX Designer": ["ux designer", "ui designer", "product designer"],
}


def _alias_pattern(alias: str) -> str:
    # Lookarounds instead of \b so aliases ending in symbols (c++, c#, .net) work.
    return rf"(?<![A-Za-z0-9+#.]){re.escape(alias)}(?![A-Za-z0-9+#])"


@lru_cache(maxsize=1)
def _compiled_skills() -> list[tuple[str, re.Pattern[str]]]:
    return [
        (skill, re.compile("|".join(_alias_pattern(a) for a in aliases), re.IGNORECASE))
        for skill, aliases in SKILL_ALIASES.items()
    ]


def extract_skills(text: str) -> list[str]:
    """Return canonical skills found in text, most frequently mentioned first."""
    counts: list[tuple[str, int]] = []
    for skill, pattern in _compiled_skills():
        n = len(pattern.findall(text))
        if n:
            counts.append((skill, n))
    counts.sort(key=lambda item: -item[1])
    return [skill for skill, _ in counts]


def normalize_skill(name: str) -> str:
    """Map a free-form skill name (e.g. from Gemini) to its canonical form if known."""
    cleaned = name.strip()
    lowered = cleaned.lower()
    for skill, aliases in SKILL_ALIASES.items():
        if lowered == skill.lower() or lowered in aliases:
            return skill
    return cleaned


def extract_roles(text: str) -> list[str]:
    """Return known role titles mentioned in text, ordered by first appearance."""
    lowered = text.lower()
    found: list[tuple[int, str]] = []
    for role in KNOWN_ROLES:
        spellings = [role.lower(), *ROLE_ALIASES.get(role, [])]
        positions = [
            m.start()
            for s in spellings
            for m in [re.search(_alias_pattern(s), lowered)]
            if m
        ]
        if positions:
            found.append((min(positions), role))
    found.sort()
    return [role for _, role in found]
