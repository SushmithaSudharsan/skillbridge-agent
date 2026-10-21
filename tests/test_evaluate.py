"""Every labelled test student must get exactly the gaps we expect."""

import pytest

from skillbridge.catalog import load_catalog
from skillbridge.evaluate import load_students, run_student

STUDENTS = load_students()


@pytest.mark.parametrize("student", STUDENTS, ids=[s.id for s in STUDENTS])
def test_student_gaps_match_labels(student):
    result = run_student(student, load_catalog())
    assert result.false_positives == 0, [g.skill for g in result.gaps]
    assert result.false_negatives == 0
    assert result.plan_problems == []
