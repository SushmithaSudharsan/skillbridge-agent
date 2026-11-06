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

Priority = Literal["required", "important"]
PRIORITY_ORDER = {"required": 0, "important": 1}
Seniority = Literal["fresher", "mid", "senior"]
Source = Literal["job", "project", "certification", "course", "self_study", "listed", "implied"]
Confidence = Literal["low", "medium", "high"]


# ---------- Lookup tables ----------


# Where the proof came from -> how sure we are. Decided by code, not the LLM.
SOURCE_CONFIDENCE: dict[str, str] = {
    "job": "high",
    "project": "high",
    "certification": "medium",
    "course": "medium",
    "self_study": "medium",
    "listed": "low",
    "implied": "medium",
}


# ---------- Catalog ----------

# ---------- Roles ----------

class RoleFamily(BaseModel):
    """One of the role families, with every job title that means it."""

    model_config = ConfigDict(frozen=True)

    name: str
    titles: list[str] = Field(min_length=1)


class TitleMatch(BaseModel):
    """A job title split into its role family and seniority."""

    model_config = ConfigDict(frozen=True)

    family: str | None             # None = not one of our role families
    seniority: Seniority | None   # None = the title doesn't say, or the words conflict


class SeniorityDecision(BaseModel):
    """Which seniority a job is, where that came from, and why."""

    model_config = ConfigDict(frozen=True)

    seniority: Seniority | None
    source: Literal["jd_years", "jd_words", "title", "jd_hints", "unknown"]
    note: str


# ---------- Student ----------

class StudentSkill(BaseModel):
    """One skill a student has, with the resume text that proves it."""

    model_config = ConfigDict(frozen=True)

    skill: str
    level: Level
    source: Source
    evidence: str = Field(min_length=1)   # exact words from the resume
    via: str | None = None                # for "implied": the skill whose evidence implies this one

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


# ---------- Plan ----------

