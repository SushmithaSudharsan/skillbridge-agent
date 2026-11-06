"""Turn the many ways people write a skill ("sklearn", "Scikit Learn") into one name."""

import difflib
import json
from pathlib import Path

from app.services.catalog import DEFAULT_DATA_DIR


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
        """Exact name or alias, then a close spelling, then a unique whole-word containment
        ("spark" -> "Apache Spark"). Ambiguous or short names are never guessed."""
        key = _norm(text)
        if key in self.lookup:
            return self.lookup[key]
        close = difflib.get_close_matches(key, list(self.lookup), n=1, cutoff=self.cutoff)
        if close:
            return self.lookup[close[0]]
        if len(key) < 4:
            return None
        hits = {skill for name, skill in self.lookup.items()
                if len(name) >= 4 and (_contains(name, key) or _contains(key, name))}
        return hits.pop() if len(hits) == 1 else None

    def extended(self, extra: list[str]) -> "SkillMatcher":
        """A copy that also knows `extra` names (e.g. this JD's requirements), so a resume's
        wording maps onto the job's wording. Known aliases still win."""
        new = SkillMatcher([], cutoff=self.cutoff)
        new.lookup = dict(self.lookup)
        for name in extra:
            new.lookup.setdefault(_norm(name), name)
        return new


def load_matcher(canonical_skills: list[str], data_dir: Path | None = None) -> SkillMatcher:
    path = (data_dir or DEFAULT_DATA_DIR) / "skill_aliases.json"
    aliases = json.loads(path.read_text(encoding="utf-8"))
    return SkillMatcher(canonical_skills, aliases)


def _contains(longer: str, shorter: str) -> bool:
    """Whole-word containment: 'apache spark' contains 'spark', 'sparkling' does not."""
    a, b = longer.split(), shorter.split()
    return len(b) < len(a) and any(a[i:i + len(b)] == b for i in range(len(a) - len(b) + 1))


def _norm(text: str) -> str:
    return " ".join(text.casefold().replace("-", " ").replace("_", " ").split())
