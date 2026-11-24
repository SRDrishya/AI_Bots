"""Semantic index of all questions: LangChain + Chroma with Gemini embeddings.

Used for
  - duplicate detection: reworded questions ("What is DI?" vs "How does dependency injection work?")
  - search by meaning in the UI ("slow queries" finds "How do you tune a query that uses a scan?")
  - related questions under each answer

One persistent Chroma collection in data/chroma; every vector carries its bank key as metadata so
searches stay inside one technology + experience bank. Chroma calls are blocking, so they run in a
worker thread. Without a Gemini key the index is disabled and the app falls back to word matching.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

from app.config import get_env_settings
from app.models import Question

log = logging.getLogger(__name__)

COLLECTION = "interview_questions"
RATE_LIMIT_WAITS = (15, 30, 60)   # seconds to wait after a 429 before retrying (Gemini per-minute limits)


def _with_retry(fn, *args):
    """Call fn, waiting and retrying when Gemini says the per-minute quota is used up."""
    for wait in (*RATE_LIMIT_WAITS, None):
        try:
            return fn(*args)
        except Exception as exc:
            text = str(exc)
            if wait is None or ("429" not in text and "RESOURCE_EXHAUSTED" not in text):
                raise
            log.info("Embedding rate limit hit; retrying in %ss", wait)
            time.sleep(wait)


class EmbeddingError(RuntimeError):
    """Embedding / vector-store failure; message is safe to show the user."""


class QuestionIndex:
    def __init__(self, persist_dir: Path, embeddings=None):
        """`embeddings`: any LangChain Embeddings (tests pass a fake one). None = Gemini from .env."""
        self.persist_dir = persist_dir
        self._embeddings = embeddings
        self._store = None
        self._lock = threading.Lock()
        self._vectors: dict[str, list[float]] = {}   # question text -> embedding, reused when indexing

    @property
    def enabled(self) -> bool:
        return self._embeddings is not None or bool(get_env_settings().gemini_api_key)

    def _vector_store(self):
        with self._lock:
            if self._store is None:
                import chromadb
                from langchain_chroma import Chroma

                embeddings = self._embeddings
                if embeddings is None:
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings

                    env = get_env_settings()
                    embeddings = GoogleGenerativeAIEmbeddings(model=env.embedding_model,
                                                              google_api_key=env.gemini_api_key)
                self._store = Chroma(
                    collection_name=COLLECTION,
                    embedding_function=embeddings,
                    persist_directory=str(self.persist_dir),
                    client_settings=chromadb.config.Settings(anonymized_telemetry=False,
                                                             is_persistent=True,
                                                             persist_directory=str(self.persist_dir)),
                    collection_configuration={"hnsw": {"space": "cosine"}},
                )
            return self._store

    async def _run(self, fn, *args):
        try:
            return await asyncio.to_thread(fn, *args)
        except Exception as exc:  # network, quota, bad key: callers fall back to word matching
            log.warning("Vector index error: %s", exc)
            raise EmbeddingError(f"Semantic index unavailable ({str(exc)[:160]})") from exc

    # ---------- writes ----------

    def _add(self, key: str, questions: list[Question]) -> None:
        store = self._vector_store()
        existing = set(store.get(where={"bank": key}, include=[])["ids"])
        new = [q for q in questions if q.id not in existing]
        if not new:
            return
        vectors = [self._vectors.get(q.question) for q in new]
        if any(v is None for v in vectors):
            vectors = self._embed([q.question for q in new])
        # Vectors were already computed for the duplicate check: add them directly instead of re-embedding.
        store._collection.upsert(ids=[q.id for q in new], documents=[q.question for q in new],
                                 embeddings=vectors,
                                 metadatas=[{"bank": key} for _ in new])
        for q in new:
            self._vectors.pop(q.question, None)

    async def add(self, key: str, questions: list[Question]) -> None:
        if questions:
            await self._run(self._add, key, questions)

    def _delete_ids(self, ids: list[str]) -> None:
        if ids:
            self._vector_store().delete(ids=ids)

    async def delete(self, ids: list[str]) -> None:
        await self._run(self._delete_ids, ids)

    def _delete_bank(self, key: str) -> None:
        store = self._vector_store()
        self._delete_ids(store.get(where={"bank": key}, include=[])["ids"])

    async def delete_bank(self, key: str) -> None:
        await self._run(self._delete_bank, key)

    # ---------- reads ----------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = _with_retry(self._vector_store().embeddings.embed_documents, texts)
        if len(self._vectors) > 5000:
            self._vectors.clear()
        self._vectors.update(zip(texts, vectors, strict=True))
        return vectors

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._run(self._embed, texts)

    def _nearest(self, key: str, vectors: list[list[float]]) -> list[tuple[str, str, float] | None]:
        """For each vector: (id, text, cosine distance) of the closest question in the bank, or None."""
        store = self._vector_store()
        if not store.get(where={"bank": key}, limit=1, include=[])["ids"]:
            return [None] * len(vectors)
        out = []
        for vector in vectors:
            # despite its name this returns the raw cosine *distance* (0 = identical)
            hits = store.similarity_search_by_vector_with_relevance_scores(vector, k=1, filter={"bank": key})
            if hits:
                doc, distance = hits[0]
                out.append((doc.id, doc.page_content, float(distance)))
            else:
                out.append(None)
        return out

    async def nearest(self, key: str, vectors: list[list[float]]):
        return await self._run(self._nearest, key, vectors)

    def _search(self, key: str, query: str, k: int) -> list[tuple[str, float]]:
        hits = _with_retry(self._vector_store().similarity_search_with_score, query, k, {"bank": key})
        return [(doc.id, round(1.0 - distance, 3)) for doc, distance in hits]  # cosine similarity

    async def search(self, key: str, query: str, k: int = 30) -> list[tuple[str, float]]:
        """(question id, cosine similarity) best match first."""
        return await self._run(self._search, key, query, k)

    def _related(self, key: str, question_id: str, k: int) -> list[tuple[str, float]]:
        store = self._vector_store()
        got = store.get(ids=[question_id], include=["embeddings"])
        if not got["ids"]:
            return []
        hits = store.similarity_search_by_vector_with_relevance_scores(got["embeddings"][0], k=k + 1,
                                                                       filter={"bank": key})
        return [(doc.id, round(1.0 - distance, 3)) for doc, distance in hits if doc.id != question_id][:k]

    async def related(self, key: str, question_id: str, k: int = 5) -> list[tuple[str, float]]:
        return await self._run(self._related, key, question_id, k)
