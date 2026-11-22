import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

DEFAULT_MODEL = "gpt-4.1-mini"


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to .env or set it in your environment."
            )

        model = os.getenv("TUTORIAL_MODEL", DEFAULT_MODEL).strip()
        if not model:
            raise ValueError("TUTORIAL_MODEL cannot be empty.")

        return cls(api_key=api_key, model=model)
