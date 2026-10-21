"""Tests for the data models in skillbridge.models."""

import pytest
from pydantic import ValidationError

from skillbridge.models import (
    Course,
    Level,
    Project,
    Requirement,
    RoleFamily,
    RoleProfile,
    SkillGap,
    StudentSkill,
)


def make_course(**overrides) -> Course:
    """Build a valid Course, replacing any fields passed in."""
    fields = {
        "course_id": "C999",
        "course_name": "Test Course",
        "skills_taught": ["SQL"],
        "prerequisites": [],
        "difficulty": "Beginner",
        "duration_weeks": 2,
        "careers": ["Data Analyst"],
    }
    fields.update(overrides)
    return Course(**fields)


# ---------- Level ----------

def test_levels_compare_like_numbers():
    assert Level.NONE < Level.BEGINNER < Level.INTERMEDIATE < Level.ADVANCED


def test_level_can_climb_one_step():
    assert Level(Level.BEGINNER + 1) == Level.INTERMEDIATE


def test_level_above_advanced_does_not_exist():
    with pytest.raises(ValueError):
        Level(Level.ADVANCED + 1)


# ---------- Course ----------

@pytest.mark.parametrize(
    "difficulty, expected",
    [
        ("Beginner", Level.BEGINNER),
        ("Intermediate", Level.INTERMEDIATE),
        ("Advanced", Level.ADVANCED),
    ],
)
def test_course_level_comes_from_difficulty(difficulty, expected):
    assert make_course(difficulty=difficulty).level == expected


def test_misspelled_difficulty_is_rejected():
    with pytest.raises(ValidationError):
        make_course(difficulty="Intermediat")


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        make_course(prerequisite=["SQL"])


def test_missing_required_field_is_rejected():
    fields = make_course().model_dump()
    del fields["course_name"]
    with pytest.raises(ValidationError):
        Course(**fields)


@pytest.mark.parametrize("weeks", [0, -1])
def test_non_positive_duration_is_rejected(weeks):
    with pytest.raises(ValidationError):
        make_course(duration_weeks=weeks)


def test_course_must_teach_at_least_one_skill():
    with pytest.raises(ValidationError):
        make_course(skills_taught=[])


def test_prerequisites_default_to_empty_list():
    fields = make_course().model_dump()
    del fields["prerequisites"]
    assert Course(**fields).prerequisites == []


def test_default_prerequisites_are_not_shared():
    fields = make_course().model_dump()
    del fields["prerequisites"]
    first, second = Course(**fields), Course(**fields)
    first.prerequisites.append("Python")
    assert second.prerequisites == []


def test_course_is_frozen():
    course = make_course()
    with pytest.raises(ValidationError):
        course.duration_weeks = 5


# ---------- Requirement ----------

def test_requirement_accepts_int_level():
    req = Requirement(skill="SQL", level=2, priority="required")
    assert req.level is Level.INTERMEDIATE


@pytest.mark.parametrize("level", [-1, 4])
def test_requirement_level_out_of_range_is_rejected(level):
    with pytest.raises(ValidationError):
        Requirement(skill="SQL", level=level, priority="required")


def test_requirement_priority_must_be_known():
    with pytest.raises(ValidationError):
        Requirement(skill="SQL", level=2, priority="nice to have")


def test_requirement_share_defaults_to_none():
    assert Requirement(skill="SQL", level=2, priority="required").share is None


@pytest.mark.parametrize("share", [0.0, 0.5, 1.0])
def test_requirement_share_within_bounds_is_accepted(share):
    req = Requirement(skill="SQL", level=2, priority="required", share=share)
    assert req.share == share


@pytest.mark.parametrize("share", [-0.1, 1.1])
def test_requirement_share_out_of_bounds_is_rejected(share):
    with pytest.raises(ValidationError):
        Requirement(skill="SQL", level=2, priority="required", share=share)


# ---------- RoleFamily and RoleProfile ----------

def test_role_family_keeps_its_titles():
    family = RoleFamily(name="ML Engineer",
                        titles=["Machine Learning Engineer", "AI/ML Engineer"])
    assert "AI/ML Engineer" in family.titles


def test_role_family_needs_at_least_one_title():
    with pytest.raises(ValidationError):
        RoleFamily(name="ML Engineer", titles=[])


def test_role_profile_accepts_known_seniority():
    profile = RoleProfile(
        family="ML Engineer", seniority="fresher",
        requirements=[Requirement(skill="Python", level=2, priority="required")],
    )
    assert profile.seniority == "fresher"


def test_role_profile_rejects_unknown_seniority():
    with pytest.raises(ValidationError):
        RoleProfile(
            family="ML Engineer", seniority="junior",
            requirements=[Requirement(skill="Python", level=2, priority="required")],
        )


def test_role_profile_needs_at_least_one_requirement():
    with pytest.raises(ValidationError):
        RoleProfile(family="ML Engineer", seniority="mid", requirements=[])


# ---------- StudentSkill ----------

@pytest.mark.parametrize(
    "source, expected",
    [
        ("job", "high"),
        ("project", "high"),
        ("certification", "medium"),
        ("course", "medium"),
        ("self_study", "medium"),
        ("listed", "low"),
    ],
)
def test_confidence_comes_from_source(source, expected):
    skill = StudentSkill(skill="Docker", level=1, source=source,
                         evidence="Docker")
    assert skill.confidence == expected


def test_skill_without_evidence_is_rejected():
    with pytest.raises(ValidationError):
        StudentSkill(skill="Kubernetes", level=1, source="listed", evidence="")


def test_unknown_source_is_rejected():
    with pytest.raises(ValidationError):
        StudentSkill(skill="Kubernetes", level=1, source="youtube",
                     evidence="watched videos")


# ---------- SkillGap ----------

@pytest.mark.parametrize(
    "current, required, expected",
    [
        (Level.NONE, Level.ADVANCED, 3),
        (Level.BEGINNER, Level.INTERMEDIATE, 1),
    ],
)
def test_gap_size(current, required, expected):
    gap = SkillGap(skill="Power BI", current=current,
                   required=required, priority="required")
    assert gap.size == expected


def test_gap_converts_ints_to_levels():
    gap = SkillGap(skill="Power BI", current=0, required=3, priority="important")
    assert gap.current is Level.NONE
    assert gap.required is Level.ADVANCED


@pytest.mark.parametrize(
    "current, required",
    [
        (Level.INTERMEDIATE, Level.INTERMEDIATE),   # already there
        (Level.ADVANCED, Level.BEGINNER),           # already above
    ],
)
def test_non_gap_is_rejected(current, required):
    with pytest.raises(ValidationError):
        SkillGap(skill="SQL", current=current, required=required,
                 priority="required")


# ---------- Project ----------

def test_project_level_comes_from_difficulty():
    project = Project(project_id="P999", project_name="Test", description="A test.",
                      skills=["SQL"], difficulty="Advanced", careers=["Data Analyst"])
    assert project.level == Level.ADVANCED


def test_project_must_use_at_least_one_skill():
    with pytest.raises(ValidationError):
        Project(project_id="P999", project_name="Test", description="A test.",
                skills=[], difficulty="Beginner", careers=["Data Analyst"])
