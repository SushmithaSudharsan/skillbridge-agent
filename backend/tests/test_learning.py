"""Learning recommendations: course areas and topics, never course IDs."""

import pytest

from app.services.catalog import load_catalog
from app.services.learning import learning_steps, load_guide
from app.services.models import Level, SkillGap


@pytest.fixture(scope="module")
def guide():
    return load_guide()


def gap(skill, cur, req, priority="required"):
    return SkillGap(skill=skill, current=cur, required=req, priority=priority)


def test_guide_covers_every_skill_at_every_level(guide):
    assert set(guide) == set(load_catalog().skills)
    for skill, info in guide.items():
        assert set(info.levels) == {1, 2, 3}, skill
        for level in info.levels.values():
            assert level.area and level.topics, skill


def test_guide_prerequisites_are_known_skills(guide):
    for skill, info in guide.items():
        assert set(info.prerequisites) <= set(guide), skill


def test_steps_climb_one_level_at_a_time(guide):
    steps = learning_steps([gap("RAG", 0, 3)], {"LLMs": Level.BEGINNER, "Python": Level.BEGINNER}, guide)
    assert [(s.skill, s.level) for s in steps] == [("RAG", 1), ("RAG", 2), ("RAG", 3)]
    assert steps[0].text == "Take a course in Fundamentals of RAG (Beginner)"
    assert "chunking documents" in steps[0].topics


def test_prerequisites_come_first(guide):
    steps = learning_steps([gap("RAG", 0, 1)], {}, guide)
    assert [s.skill for s in steps] == ["Python", "LLMs", "RAG"]
    assert steps[0].reason == "needed before LLMs"


def test_steps_never_mention_course_ids(guide):
    steps = learning_steps([gap("AI Agents", 0, 2), gap("Docker", 0, 2)], {}, guide)
    for s in steps:
        assert not any(ch.isdigit() for ch in s.area), s.area


def test_skill_without_a_guide_still_gets_a_step(guide):
    steps = learning_steps([gap("Kubernetes", 0, 2)], {}, guide)
    assert steps[0].skill == "Kubernetes" and not steps[0].in_guide and steps[0].topics == []


def test_no_step_is_repeated(guide):
    steps = learning_steps([gap("RAG", 0, 1), gap("AI Agents", 0, 1)], {}, guide)
    keys = [(s.skill, s.level) for s in steps]
    assert len(keys) == len(set(keys))
