"""Run every test student through gaps + planner and score the results.

Run it with:  python -m skillbridge.evaluate
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter

from skillbridge.catalog import DEFAULT_DATA_DIR, Catalog, load_catalog
from skillbridge.experience import decide_seniority
from skillbridge.gaps import find_gaps, student_levels
from skillbridge.models import Level, PlanResult, SkillGap, StudentSkill
from skillbridge.planner import make_plan, plan_problems


class ExpectedGap(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill: str
    current: Level
    required: Level


class TestStudent(BaseModel):
    """One labelled example: a student, a target job, and the gaps a human expects."""

    __test__ = False          # tells pytest this is not a test class
    model_config = ConfigDict(frozen=True)

    id: str
    description: str
    target_title: str
    jd_text: str = ""
    skills: list[StudentSkill]
    expected_gaps: list[ExpectedGap]


class StudentResult(BaseModel):
    student_id: str
    family: str
    seniority: str
    gaps: list[SkillGap]
    plan: PlanResult
    true_positives: int
    false_positives: int
    false_negatives: int
    plan_problems: list[str]


def load_students(path: Path | None = None) -> list[TestStudent]:
    path = path or DEFAULT_DATA_DIR / "test_students.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TypeAdapter(list[TestStudent]).validate_python(raw)


def run_student(student: TestStudent, catalog: Catalog) -> StudentResult:
    match = catalog.parse_title(student.target_title)
    if match is None:
        raise ValueError(f"{student.id}: unsupported title '{student.target_title}'")
    decision = decide_seniority(match.seniority, student.jd_text)
    if decision.seniority is None:
        raise ValueError(f"{student.id}: seniority unknown ({decision.note})")

    profile = catalog.profile(match.family, decision.seniority)
    levels = student_levels(student.skills)
    gaps = find_gaps(levels, profile)
    plan = make_plan(gaps, levels, catalog, match.family)

    # A gap counts as correct only if both the skill AND the levels match.
    predicted = {(g.skill, g.current, g.required) for g in gaps}
    expected = {(g.skill, g.current, g.required) for g in student.expected_gaps}
    return StudentResult(
        student_id=student.id, family=match.family, seniority=decision.seniority,
        gaps=gaps, plan=plan,
        true_positives=len(predicted & expected),
        false_positives=len(predicted - expected),
        false_negatives=len(expected - predicted),
        plan_problems=plan_problems(plan.course_ids, levels, catalog),
    )


def ratio(top: int, bottom: int) -> float:
    """top / bottom, but 1.0 when there is nothing to count (no gaps expected or found)."""
    return top / bottom if bottom else 1.0


def main() -> None:
    catalog = load_catalog()
    results = [run_student(s, catalog) for s in load_students()]

    print(f"{'student':<22} {'role':<28} {'gaps':>4} {'prec':>5} {'rec':>5} {'weeks':>5}  valid")
    for r in results:
        precision = ratio(r.true_positives, r.true_positives + r.false_positives)
        recall = ratio(r.true_positives, r.true_positives + r.false_negatives)
        role = f"{r.family} / {r.seniority}"
        print(f"{r.student_id:<22} {role:<28} {len(r.gaps):>4} {precision:>5.2f} {recall:>5.2f} "
              f"{r.plan.total_weeks:>5}  {'yes' if not r.plan_problems else 'NO'}")

    tp = sum(r.true_positives for r in results)
    fp = sum(r.false_positives for r in results)
    fn = sum(r.false_negatives for r in results)
    valid = sum(1 for r in results if not r.plan_problems)
    print()
    print(f"Gap precision (micro): {ratio(tp, tp + fp):.2f}")
    print(f"Gap recall    (micro): {ratio(tp, tp + fn):.2f}")
    print(f"Path validity:         {valid}/{len(results)}")
    print()
    for r in results:
        print(f"{r.student_id}: " + " -> ".join(r.plan.course_ids or ["(no courses needed)"]))


if __name__ == "__main__":
    main()
