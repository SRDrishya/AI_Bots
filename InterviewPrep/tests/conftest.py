import hashlib
import math
import re
from typing import ClassVar

import pytest
from langchain_core.embeddings import Embeddings


@pytest.fixture(autouse=True)
def no_real_ai(monkeypatch):
    """Tests never call real AI providers: blank keys win over the values in .env."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")


class BagOfWordsEmbeddings(Embeddings):
    """Offline stand-in for Gemini embeddings: texts sharing words get close vectors."""

    STOP = frozenset({"what", "is", "the", "a", "an", "how", "do", "does", "you", "in", "of", "and", "to", "why", "explain", "it"})

    # synonyms make reworded questions land close together, like real embeddings do
    SYNONYMS: ClassVar[dict[str, str]] = {"di": "dependency injection", "dotnet": "net", "2nd": "second", "top": "highest", "pay": "salary",
                "get": "find"}

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * 128
        words = " ".join(self.SYNONYMS.get(w, w) for w in re.findall(r"[a-z0-9#]+", text.lower())).split()
        for word in words:
            if word not in self.STOP:
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 128] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)
