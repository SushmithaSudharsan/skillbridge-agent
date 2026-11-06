"""Prompts live in /prompts as versioned Markdown files, one per prompt.

Each file starts with a small header:
    ---
    version: 3
    ---
Bump the version whenever you change the wording. Every LLM call is logged with the
prompt-set version (see core/tracing.py), so a quality change can be traced to a prompt change.
"""

import hashlib
import os
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(os.environ.get("SKILLBRIDGE_PROMPTS_DIR", Path(__file__).resolve().parents[3] / "prompts"))


@lru_cache(maxsize=None)
def _load(name: str) -> tuple[int, str]:
    raw = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise ValueError(f"prompts/{name}.md must start with a '---' header")
    header, body = raw[4:].split("\n---\n", 1)
    meta = dict(line.split(":", 1) for line in header.splitlines() if ":" in line)
    return int(meta["version"].strip()), body.rstrip("\n")


def text(name: str) -> str:
    return _load(name)[1]


def version(name: str) -> int:
    return _load(name)[0]


@lru_cache(maxsize=1)
def set_version() -> str:
    """One short id for the whole prompt set: changes whenever any prompt file changes."""
    digest = hashlib.sha256()
    for path in sorted(PROMPTS_DIR.glob("*.md")):
        digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()[:8]


# Personas, sent as system messages
READER = text("system_reader")
ADVISOR = text("system_advisor")
JUDGE = text("system_judge")
DEBATER = text("system_debater")
ANSWER_READER = text("system_answer_reader")
