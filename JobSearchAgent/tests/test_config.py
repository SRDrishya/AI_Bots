from __future__ import annotations

import re

from app.config import BASE_DIR

SECRET_NAMES = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|APP_ID)$")


def test_env_example_contains_no_secrets():
    """.env.example is meant to be shared — every key-like setting must be blank."""
    leaked = []
    for line in (BASE_DIR / ".env.example").read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        if SECRET_NAMES.search(name) and value:
            leaked.append(name)
    assert not leaked, f".env.example must not contain real values for: {', '.join(leaked)}"


def test_env_is_gitignored():
    ignored = (BASE_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored and "data/" in ignored
