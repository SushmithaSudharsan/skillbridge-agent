"""Turn the many ways people write a skill ("sklearn", "Scikit Learn") into one name."""

import difflib
import json
from pathlib import Path

from skillbridge.catalog import DEFAULT_DATA_DIR


class SkillMatcher:
    """Exact alias first, then a close spelling match. Never guesses beyond that."""

    def __init__(self, canonical_skills: list[str], aliases: dict[str, list[str]] | None = None,
                 cutoff: float = 0.88):
        self.lookup: dict[str, str] = {}
        for skill in canonical_skills:
            self.lookup[_norm(skill)] = skill
        for skill, names in (aliases or {}).items():
            if skill not in canonical_skills:
                raise ValueError(f"Alias target '{skill}' is not a known skill")
            for name in names:
                self.lookup[_norm(name)] = skill
        self.cutoff = cutoff

    def match(self, text: str) -> str | None:
        key = _norm(text)
        if key in self.lookup:
            return self.lookup[key]
        close = difflib.get_close_matches(key, list(self.lookup), n=1, cutoff=self.cutoff)
        return self.lookup[close[0]] if close else None


def load_matcher(canonical_skills: list[str], data_dir: Path | None = None) -> SkillMatcher:
    path = (data_dir or DEFAULT_DATA_DIR) / "skill_aliases.json"
    aliases = json.loads(path.read_text(encoding="utf-8"))
    return SkillMatcher(canonical_skills, aliases)


def _norm(text: str) -> str:
    return " ".join(text.casefold().replace("-", " ").replace("_", " ").split())
