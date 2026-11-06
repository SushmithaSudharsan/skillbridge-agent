"""What to learn next, as course AREAS with key topics (not specific courses).

Steps climb one level at a time and put prerequisites first, using the same rules as
the M1 planner, but they recommend a field of study instead of a course ID.
"""

import json
from pathlib import Path

from pydantic import BaseModel

from app.services.catalog import DEFAULT_DATA_DIR
from app.services.models import Level, SkillGap

LEVEL_NAMES = ["None", "Beginner", "Intermediate", "Advanced"]


class LevelGuide(BaseModel):
    area: str
    topics: list[str]


class SkillGuide(BaseModel):
    prerequisites: list[str]
    levels: dict[int, LevelGuide]


class LearningStep(BaseModel):
    skill: str
    level: Level
    area: str               # e.g. "Fundamentals of RAG"
    topics: list[str]
    reason: str             # why this step is here
    in_guide: bool = True   # False: no curated guide; area/topics may be AI-suggested
    suggested: bool = False # True: area/topics written by the LLM (checked), not curated

    @property
    def text(self) -> str:
        return f"Take a course in {self.area} ({LEVEL_NAMES[self.level]})"


def load_guide(path: Path | None = None) -> dict[str, SkillGuide]:
    raw = json.loads((path or DEFAULT_DATA_DIR / "learning_guide.json").read_text(encoding="utf-8"))
    return {skill: SkillGuide.model_validate(value) for skill, value in raw.items()}


def learning_steps(gaps: list[SkillGap], levels: dict[str, Level],
                   guide: dict[str, SkillGuide]) -> list[LearningStep]:
    current = dict(levels)
    steps: list[LearningStep] = []

    def climb(skill: str, target: Level, reason: str, visiting: frozenset[str]) -> None:
        if skill in visiting:
            return                                          # prerequisite loop: stop quietly
        info = guide.get(skill)
        while current.get(skill, Level.NONE) < target:
            nxt = Level(current.get(skill, Level.NONE) + 1)
            if info is None:                                # a JD skill we have no guide for
                steps.append(LearningStep(skill=skill, level=target, in_guide=False,
                                          area=f"{skill}", topics=[], reason=reason))
                current[skill] = target
                return
            if nxt == Level.BEGINNER:
                for pre in sorted(info.prerequisites):
                    if current.get(pre, Level.NONE) < Level.BEGINNER:
                        climb(pre, Level.BEGINNER, f"needed before {skill}", visiting | {skill})
            step = info.levels[int(nxt)]
            steps.append(LearningStep(skill=skill, level=nxt, area=step.area,
                                      topics=step.topics, reason=reason))
            current[skill] = nxt

    for gap in gaps:
        why = "required by the job" if gap.priority == "required" else "a plus for the job"
        climb(gap.skill, gap.required, why, frozenset())
    return steps
