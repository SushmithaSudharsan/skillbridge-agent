"""Related skills: PyTorch <-> TensorFlow, Power BI <-> Tableau, MySQL <-> PostgreSQL...

A related skill never counts as meeting a requirement. It changes how we talk about the gap:
"you have the closest thing, show how it transfers" instead of "you have nothing".
"""

import json
from functools import lru_cache

from app.services.catalog import DEFAULT_DATA_DIR


@lru_cache
def _groups() -> list[list[str]]:
    return json.loads((DEFAULT_DATA_DIR / "related_skills.json").read_text(encoding="utf-8"))["groups"]


def peers(skill: str) -> list[str]:
    """Other tools in the same family as `skill` (empty if none)."""
    key = skill.casefold()
    for group in _groups():
        names = [g.casefold() for g in group]
        if key in names:
            return [g for g in group if g.casefold() != key]
    return []
