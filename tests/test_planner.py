"""Tests for the learning-path planner, using the real catalog."""

import os
import subprocess
import sys

import pytest

from skillbridge.catalog import load_catalog
from skillbridge.gaps import find_gaps
from skillbridge.models import Level
from skillbridge.planner import make_plan, pick_course, plan_problems


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def plan_for(catalog, levels, family, seniority):
    gaps = find_gaps(levels, catalog.profile(family, seniority))
    return make_plan(gaps, levels, catalog, family)


def test_student_without_python_starts_with_fundamentals(catalog):
    plan = plan_for(catalog, {"Excel": Level.BEGINNER}, "Data Analyst", "fresher")
    ids = plan.course_ids
    assert ids.index("C001") < ids.index("C010")    # Python basics before Pandas
    assert plan.uncovered == []


def test_every_planned_course_has_its_prerequisites(catalog):
    for family in ["Data Analyst", "Data Scientist", "ML Engineer", "GenAI Engineer", "MLOps Engineer"]:
        for seniority in ["fresher", "mid", "senior"]:
            plan = plan_for(catalog, {}, family, seniority)
            assert plan_problems(plan.course_ids, {}, catalog) == [], (family, seniority)
            assert plan.uncovered == [], (family, seniority)


def test_climbs_one_level_at_a_time(catalog):
    plan = plan_for(catalog, {}, "GenAI Engineer", "senior")
    ids = plan.course_ids
    assert ids.index("C019") < ids.index("C117") < ids.index("C214")   # RAG 1 -> 2 -> 3


def test_no_course_is_repeated(catalog):
    plan = plan_for(catalog, {}, "ML Engineer", "senior")
    assert len(plan.course_ids) == len(set(plan.course_ids))


def test_student_who_meets_everything_gets_no_courses(catalog):
    profile = catalog.profile("Data Analyst", "fresher")
    levels = {req.skill: req.level for req in profile.requirements}
    assert plan_for(catalog, levels, "Data Analyst", "fresher").courses == []


def test_pick_course_prefers_the_role_family(catalog):
    course = pick_course(catalog, "Python", Level.INTERMEDIATE, "GenAI Engineer")
    assert "GenAI Engineer" in course.careers


def test_plan_problems_catch_a_wrong_order(catalog):
    problems = plan_problems(["C010", "C001"], {}, catalog)     # Pandas before Python
    assert any("before Python" in p for p in problems)


def test_plan_problems_catch_an_invented_course(catalog):
    assert plan_problems(["C999"], {}, catalog) == ["C999 is not in the catalog"]


def test_same_answer_every_run():
    """Run the evaluation in fresh Python processes with different hash seeds."""
    outputs = set()
    for seed in ["1", "2", "3"]:
        result = subprocess.run(
            [sys.executable, "-m", "skillbridge.evaluate"],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True, text=True, check=True,
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1
