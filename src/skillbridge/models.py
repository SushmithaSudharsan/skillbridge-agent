"""Data models shared by every part of SkillBridge."""

from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------- The skill ladder ----------

class Level(IntEnum):
    """Which step of the skill ladder someone is on."""

    NONE = 0          # never used it
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3


# ---------- Allowed words ----------

Difficulty = Literal["Beginner", "Intermediate", "Advanced"]
Priority = Literal["required", "important"]
Seniority = Literal["fresher", "mid", "senior"]
Source = Literal["job", "project", "certification", "course", "self_study", "listed"]
Confidence = Literal["low", "medium", "high"]


# ---------- Lookup tables ----------

DIFFICULTY_TO_LEVEL: dict[str, Level] = {
    "Beginner": Level.BEGINNER,
    "Intermediate": Level.INTERMEDIATE,
    "Advanced": Level.ADVANCED,
}

# Where the proof came from -> how sure we are. Decided by code, not the LLM.
SOURCE_CONFIDENCE: dict[str, str] = {
    "job": "high",
    "project": "high",
    "certification": "medium",
    "course": "medium",
    "self_study": "medium",
    "listed": "low",
}


# ---------- Catalog ----------

class Course(BaseModel):
    """One course in the catalog."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    course_id: str
    course_name: str
    skills_taught: list[str] = Field(min_length=1)
    prerequisites: list[str] = []
    difficulty: Difficulty
    duration_weeks: int = Field(gt=0)
    careers: list[str]

    @property
    def level(self) -> Level:
        return DIFFICULTY_TO_LEVEL[self.difficulty]


class Project(BaseModel):
    """One portfolio project a student can build to show skills."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_id: str
    project_name: str
    description: str
    skills: list[str] = Field(min_length=1)
    difficulty: Difficulty
    careers: list[str]

    @property
    def level(self) -> Level:
        return DIFFICULTY_TO_LEVEL[self.difficulty]


# ---------- Roles ----------

class Requirement(BaseModel):
    """One skill a role needs, at a level."""

    model_config = ConfigDict(frozen=True)

    skill: str
    level: Level
    priority: Priority
    share: float | None = Field(default=None, ge=0, le=1)   # filled from job postings in M3


class RoleFamily(BaseModel):
    """One of the role families, with every job title that means it."""

    model_config = ConfigDict(frozen=True)

    name: str
    titles: list[str] = Field(min_length=1)


class RoleProfile(BaseModel):
    """What one role family needs at one seniority level."""

    model_config = ConfigDict(frozen=True)

    family: str
    seniority: Seniority
    requirements: list[Requirement] = Field(min_length=1)


# ---------- Student ----------

class StudentSkill(BaseModel):
    """One skill a student has, with the resume text that proves it."""

    model_config = ConfigDict(frozen=True)

    skill: str
    level: Level
    source: Source
    evidence: str = Field(min_length=1)   # exact words from the resume

    @property
    def confidence(self) -> Confidence:
        return SOURCE_CONFIDENCE[self.source]


# ---------- Gap ----------

class SkillGap(BaseModel):
    """How many steps a student must climb in one skill."""

    model_config = ConfigDict(frozen=True)

    skill: str
    current: Level
    required: Level
    priority: Priority

    @model_validator(mode="after")
    def required_must_exceed_current(self) -> "SkillGap":
        if self.required <= self.current:
            raise ValueError(
                f"{self.skill}: required {self.required.name} must be above "
                f"current {self.current.name}, otherwise it is not a gap"
            )
        return self

    @property
    def size(self) -> int:
        return self.required - self.current
