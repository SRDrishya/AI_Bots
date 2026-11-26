"""Local JSON storage under data/:

    banks/<slug>--<experience>.json   one question bank per technology and experience level
    chroma/              LangChain + Chroma vector index of all questions (semantic search)
    technologies.json    custom technologies added by the user
    progress.json        practice status + attempts, keyed by question id
    resume.txt           resume text used for behavioral answers and "your angle" notes
    preferences.json     target companies (searched first) and the AI provider chosen in Settings
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import TypeAdapter

from app.config import DEFAULT_TARGET_COMPANIES, JOBSEARCH_RESUME_FILE
from app.models import Bank, QuestionProgress, Technology

log = logging.getLogger(__name__)

_progress_adapter = TypeAdapter(dict[str, QuestionProgress])
_tech_adapter = TypeAdapter(list[Technology])


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Storage:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.banks_dir = data_dir / "banks"
        self.banks_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ---------- helpers ----------

    def _write(self, path: Path, text: str) -> None:
        """Write via a temp file so a crash never leaves half a JSON file behind."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def _read_json(self, path: Path, default):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except (OSError, ValueError) as exc:
            log.warning("Could not read %s (%s); starting empty", path.name, exc)
            return default

    # ---------- banks ----------

    def _bank_path(self, key: str) -> Path:
        return self.banks_dir / f"{key}.json"

    def load_bank(self, key: str) -> Bank | None:
        with self._lock:
            data = self._read_json(self._bank_path(key), None)
        if data is None:
            return None
        try:
            return Bank.model_validate(data)
        except ValueError as exc:
            log.warning("Bank %s is invalid (%s)", key, exc)
            return None

    def save_bank(self, bank: Bank) -> None:
        bank.updated_at = now_iso()
        with self._lock:
            self._write(self._bank_path(bank.key), bank.model_dump_json(indent=1))

    def delete_bank(self, key: str) -> bool:
        with self._lock:
            path = self._bank_path(key)
            if path.exists():
                path.unlink()
                return True
        return False

    def bank_counts(self) -> dict[str, int]:
        """{bank key: number of questions} for every saved bank (cheap enough for a few dozen banks)."""
        counts = {}
        for path in self.banks_dir.glob("*.json"):
            data = self._read_json(path, {})
            counts[path.stem] = len(data.get("questions", [])) if isinstance(data, dict) else 0
        return counts

    # ---------- technologies ----------

    def load_custom_technologies(self) -> list[Technology]:
        with self._lock:
            data = self._read_json(self.data_dir / "technologies.json", [])
        try:
            return _tech_adapter.validate_python(data)
        except ValueError:
            return []

    def save_custom_technologies(self, techs: list[Technology]) -> None:
        with self._lock:
            self._write(self.data_dir / "technologies.json",
                        _tech_adapter.dump_json(techs, indent=1).decode("utf-8"))

    # ---------- practice progress ----------

    def load_progress(self) -> dict[str, QuestionProgress]:
        with self._lock:
            data = self._read_json(self.data_dir / "progress.json", {})
        try:
            return _progress_adapter.validate_python(data)
        except ValueError:
            return {}

    def save_progress(self, progress: dict[str, QuestionProgress]) -> None:
        with self._lock:
            self._write(self.data_dir / "progress.json",
                        _progress_adapter.dump_json(progress, indent=1).decode("utf-8"))

    def update_progress(self, question_id: str, update) -> QuestionProgress:
        """Load, apply `update(entry)`, save — under one lock so concurrent requests don't lose writes."""
        with self._lock:
            progress = self.load_progress()
            entry = progress.get(question_id) or QuestionProgress()
            update(entry)
            entry.updated_at = now_iso()
            progress[question_id] = entry
            self.save_progress(progress)
            return entry

    # ---------- preferences ----------

    def _preferences(self) -> dict:
        data = self._read_json(self.data_dir / "preferences.json", {})
        return data if isinstance(data, dict) else {}

    def _save_preference(self, name: str, value) -> None:
        with self._lock:
            prefs = self._preferences()
            prefs[name] = value
            self._write(self.data_dir / "preferences.json", json.dumps(prefs, indent=1))

    def load_companies(self) -> list[str]:
        with self._lock:
            companies = self._preferences().get("companies")
        return companies if isinstance(companies, list) else list(DEFAULT_TARGET_COMPANIES)

    def save_companies(self, companies: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(c.strip() for c in companies if c.strip()))
        self._save_preference("companies", cleaned)
        return cleaned

    def load_provider(self) -> str:
        with self._lock:
            provider = self._preferences().get("provider")
        return provider if isinstance(provider, str) else "auto"

    def save_provider(self, provider: str) -> None:
        self._save_preference("provider", provider)

    # ---------- resume ----------

    def load_resume(self) -> str:
        path = self.data_dir / "resume.txt"
        with self._lock:
            if path.exists():
                return path.read_text(encoding="utf-8")
            imported = _import_jobsearch_resume()
            if imported:
                self._write(path, imported)
                log.info("Imported resume text from JobSearchAgent")
            return imported

    def save_resume(self, text: str) -> None:
        with self._lock:
            self._write(self.data_dir / "resume.txt", text.strip())


def _import_jobsearch_resume() -> str:
    try:
        data = json.loads(JOBSEARCH_RESUME_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    text = data.get("text", "") if isinstance(data, dict) else data if isinstance(data, str) else ""
    return text.strip()
