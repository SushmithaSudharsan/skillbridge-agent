"""Reference data: role families, seniority words and the skill vocabulary.

The vocabulary is only used to give skills consistent names ("sklearn" -> "Scikit-learn")
and to find a curated learning guide. Skills outside it still count (open vocabulary).
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter

from app.services.models import RoleFamily, Seniority, TitleMatch

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SENIORITIES = ("fresher", "mid", "senior")


class Catalog(BaseModel):
    model_config = ConfigDict(frozen=True)

    families: list[RoleFamily]
    seniority_words: dict[Seniority, list[str]]
    known_skills: list[str]          # skills with a curated learning guide or aliases

    @property
    def skills(self) -> list[str]:
        return self.known_skills

    def family_for_title(self, title: str) -> RoleFamily | None:
        wanted = title.strip().casefold()
        for family in self.families:
            if wanted in (name.casefold() for name in [family.name, *family.titles]):
                return family
        return None

    def parse_title(self, title: str) -> TitleMatch:
        """'Senior LLM Engineer' -> family + seniority. Works for any title: an unknown role
        still yields its seniority words."""
        text = title.casefold().replace("entry level", "entry-level").replace("mid level", "mid-level")
        words = text.split()
        found: set[str] = set()
        while words and (level := self._seniority_of(words[0])):
            found.add(level)
            words.pop(0)
        while words and (level := self._seniority_of(words[-1])):
            found.add(level)
            words.pop()
        family = self.family_for_title(" ".join(words))
        return TitleMatch(family=family.name if family else None,
                          seniority=found.pop() if len(found) == 1 else None)

    def _seniority_of(self, word: str) -> str | None:
        word = word.strip(".,()")
        for seniority, level_words in self.seniority_words.items():
            if word in (w.casefold() for w in level_words):
                return seniority
        return None


def load_catalog(data_dir: Path | None = None) -> Catalog:
    folder = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    guide = json.loads((folder / "learning_guide.json").read_text(encoding="utf-8"))
    aliases = json.loads((folder / "skill_aliases.json").read_text(encoding="utf-8"))
    return Catalog(
        families=TypeAdapter(list[RoleFamily]).validate_json((folder / "roles.json").read_bytes()),
        seniority_words=TypeAdapter(dict[Seniority, list[str]]).validate_json(
            (folder / "seniority_words.json").read_bytes()),
        known_skills=sorted(set(guide) | set(aliases)),
    )
