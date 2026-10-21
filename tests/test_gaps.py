"""Tests for gap analysis."""

from skillbridge.gaps import find_gaps, student_levels
from skillbridge.models import Level, Requirement, RoleProfile, StudentSkill


def profile(*requirements: tuple[str, int, str]) -> RoleProfile:
    return RoleProfile(
        family="Data Analyst", seniority="fresher",
        requirements=[Requirement(skill=s, level=l, priority=p) for s, l, p in requirements],
    )


def test_missing_skill_is_level_none():
    gaps = find_gaps({}, profile(("Power BI", 1, "required")))
    assert gaps[0].current == Level.NONE
    assert gaps[0].size == 1


def test_skill_at_required_level_is_not_a_gap():
    gaps = find_gaps({"SQL": Level.INTERMEDIATE}, profile(("SQL", 2, "required")))
    assert gaps == []


def test_skill_above_required_level_is_not_a_gap():
    gaps = find_gaps({"SQL": Level.ADVANCED}, profile(("SQL", 2, "required")))
    assert gaps == []


def test_basic_skill_still_has_a_gap_to_intermediate():
    gaps = find_gaps({"Statistics": Level.BEGINNER}, profile(("Statistics", 2, "required")))
    assert [(g.skill, g.current, g.required) for g in gaps] == [
        ("Statistics", Level.BEGINNER, Level.INTERMEDIATE)
    ]


def test_order_is_required_first_then_bigger_gap_then_name():
    gaps = find_gaps({}, profile(
        ("Pandas", 1, "important"),
        ("SQL", 1, "required"),
        ("Excel", 1, "required"),
        ("Power BI", 3, "required"),
    ))
    assert [g.skill for g in gaps] == ["Power BI", "Excel", "SQL", "Pandas"]


def test_student_levels_keep_the_highest_mention():
    skills = [
        StudentSkill(skill="Python", level=1, source="listed", evidence="Python"),
        StudentSkill(skill="Python", level=2, source="project", evidence="Built X in Python"),
    ]
    assert student_levels(skills) == {"Python": Level.INTERMEDIATE}
