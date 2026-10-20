"""Load the data files and check that they agree with each other."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from skillbridge.models import (
    Course,
    Level,
    Project,
    RoleFamily,
    RoleProfile,
    Seniority,
    TitleMatch,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SENIORITIES = ("fresher", "mid", "senior")


class CatalogError(Exception):
    """The data files are broken or disagree with each other."""


class Catalog(BaseModel):
    """Everything SkillBridge knows about roles, courses and projects."""

    model_config = ConfigDict(frozen=True)

    families: list[RoleFamily]
    profiles: list[RoleProfile]
    courses: list[Course]
    projects: list[Project]
    seniority_words: dict[Seniority, list[str]]

    @property
    def skills(self) -> list[str]:
        """Every skill some course teaches, in alphabetical order."""
        return sorted({skill for course in self.courses for skill in course.skills_taught})

    def course(self, course_id: str) -> Course | None:
        for course in self.courses:
            if course.course_id == course_id:
                return course
        return None

    def profile(self, family: str, seniority: str) -> RoleProfile | None:
        for profile in self.profiles:
            if profile.family == family and profile.seniority == seniority:
                return profile
        return None

    def family_for_title(self, title: str) -> RoleFamily | None:
        """Find the role family for a job title, ignoring case and extra spaces."""
        wanted = title.strip().casefold()
        for family in self.families:
            names = [family.name, *family.titles]
            if wanted in (name.casefold() for name in names):
                return family
        return None

    def parse_title(self, title: str) -> TitleMatch | None:
        """Split a title like 'Senior LLM Engineer' into family and seniority.

        Returns None if the job title is not one we support.
        """
        text = title.casefold().replace("entry level", "entry-level").replace("mid level", "mid-level")
        words = text.split()
        found: set[str] = set()

        # Take level words off the front: "Senior", "Junior", "Lead", ...
        while words:
            seniority = self._seniority_of(words[0])
            if seniority is None:
                break
            found.add(seniority)
            words.pop(0)

        # Take level words off the end: "I", "II", "III"
        while words:
            seniority = self._seniority_of(words[-1])
            if seniority is None:
                break
            found.add(seniority)
            words.pop()

        family = self.family_for_title(" ".join(words))
        if family is None:
            return None
        # Exactly one level found = use it. None, or conflicting words = unknown.
        return TitleMatch(family=family.name, seniority=found.pop() if len(found) == 1 else None)

    def _seniority_of(self, word: str) -> str | None:
        """Which seniority a single word means, e.g. 'Sr.' -> 'senior'."""
        word = word.strip(".,()")
        for seniority, level_words in self.seniority_words.items():
            if word in (w.casefold() for w in level_words):
                return seniority
        return None


def load_catalog(data_dir: Path | None = None) -> Catalog:
    """Read all data files, validate them, and check they agree."""
    folder = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    catalog = Catalog(
        families=_read(folder / "roles.json", list[RoleFamily]),
        profiles=_read(folder / "role_profiles.json", list[RoleProfile]),
        courses=_read(folder / "courses.json", list[Course]),
        projects=_read(folder / "portfolio_projects.json", list[Project]),
        seniority_words=_read(folder / "seniority_words.json", dict[Seniority, list[str]]),
    )
    problems = check_catalog(catalog)
    if problems:
        raise CatalogError("Catalog checks failed:\n- " + "\n- ".join(problems))
    return catalog


def _read(path: Path, shape):
    """Read one JSON file and check it has the given shape, e.g. list[Course]."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CatalogError(f"Missing data file: {path}") from None
    except json.JSONDecodeError as error:
        raise CatalogError(f"{path.name} is not valid JSON: {error}") from None
    try:
        return TypeAdapter(shape).validate_python(raw)
    except ValidationError as error:
        raise CatalogError(f"{path.name} has invalid entries:\n{error}") from None


def check_catalog(catalog: Catalog) -> list[str]:
    """Return every way the files disagree with each other. Empty list = all good."""
    problems: list[str] = []
    family_names = {family.name for family in catalog.families}
    known_skills = set(catalog.skills)

    # 1. Nothing listed twice
    for name in _duplicates([family.name for family in catalog.families]):
        problems.append(f"Role family listed twice: {name}")
    for title in _duplicates([t.casefold() for f in catalog.families for t in f.titles]):
        problems.append(f"Job title used by more than one family: {title}")
    for course_id in _duplicates([course.course_id for course in catalog.courses]):
        problems.append(f"Course ID used twice: {course_id}")
    for project_id in _duplicates([project.project_id for project in catalog.projects]):
        problems.append(f"Project ID used twice: {project_id}")

    # 2. Courses point at real careers, and their prerequisites can be learned
    for course in catalog.courses:
        for career in course.careers:
            if career not in family_names:
                problems.append(f"{course.course_id} lists unknown career '{career}'")
        for skill in course.prerequisites:
            if not _has_course(catalog, skill, Level.BEGINNER):
                problems.append(
                    f"{course.course_id} needs '{skill}' but no Beginner course teaches it"
                )

    # 3. Projects point at real careers and real skills
    for project in catalog.projects:
        for career in project.careers:
            if career not in family_names:
                problems.append(f"{project.project_id} lists unknown career '{career}'")
        for skill in project.skills:
            if skill not in known_skills:
                problems.append(f"{project.project_id} uses unknown skill '{skill}'")

    # 4. Exactly one profile for every family and seniority
    for family in sorted(family_names):
        for seniority in SENIORITIES:
            count = sum(
                1 for p in catalog.profiles if p.family == family and p.seniority == seniority
            )
            if count == 0:
                problems.append(f"Missing profile: {family} / {seniority}")
            elif count > 1:
                problems.append(f"Profile listed {count} times: {family} / {seniority}")

    # 5. Every requirement can actually be reached, one step at a time
    for profile in catalog.profiles:
        label = f"{profile.family} / {profile.seniority}"
        if profile.family not in family_names:
            problems.append(f"Profile for unknown family: {label}")
        for skill in _duplicates([req.skill for req in profile.requirements]):
            problems.append(f"{label} lists {skill} twice")
        for req in profile.requirements:
            for step in range(Level.BEGINNER, req.level + 1):
                if not _has_course(catalog, req.skill, Level(step)):
                    problems.append(
                        f"{label}: no {Level(step).name.title()} course for {req.skill}"
                    )

    # 6. Seniority words: all three levels present, no word in two levels
    for seniority in SENIORITIES:
        if not catalog.seniority_words.get(seniority):
            problems.append(f"No seniority words for: {seniority}")
    all_words = [w.casefold() for words in catalog.seniority_words.values() for w in words]
    for word in _duplicates(all_words):
        problems.append(f"Seniority word used for more than one level: {word}")

    return problems


def _has_course(catalog: Catalog, skill: str, level: Level) -> bool:
    """Is there a course that teaches `skill` at exactly `level`?"""
    return any(skill in c.skills_taught and c.level == level for c in catalog.courses)


def _duplicates(values: list[str]) -> list[str]:
    """Values that appear more than once, in alphabetical order."""
    return sorted({value for value in values if values.count(value) > 1})
