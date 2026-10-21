"""Turn skill gaps into an ordered list of courses."""

from skillbridge.catalog import Catalog
from skillbridge.models import Course, Level, PlannedCourse, PlanResult, SkillGap


def pick_course(catalog: Catalog, skill: str, level: Level, family: str) -> Course | None:
    """The best course that teaches `skill` at exactly `level`.

    Preference: made for this role family, then shorter, then lowest ID
    (so the answer is the same every time).
    """
    options = [c for c in catalog.courses if skill in c.skills_taught and c.level == level]
    options.sort(key=lambda c: (family not in c.careers, c.duration_weeks, c.course_id))
    return options[0] if options else None


def make_plan(gaps: list[SkillGap], levels: dict[str, Level],
              catalog: Catalog, family: str) -> PlanResult:
    """Climb every gap one level at a time, taking prerequisites first."""
    current = dict(levels)            # a copy: we simulate the student learning
    path: list[PlannedCourse] = []
    uncovered: list[SkillGap] = []
    warnings: list[str] = []

    def climb(skill: str, target: Level, visiting: frozenset[str], why: str) -> bool:
        """Raise `skill` to `target`, one level at a time."""
        while current.get(skill, Level.NONE) < target:
            next_level = Level(current.get(skill, Level.NONE) + 1)
            course = pick_course(catalog, skill, next_level, family)
            if course is None:
                warnings.append(f"No {next_level.name.title()} course for {skill}")
                return False
            if not take(course, visiting, f"{skill} to {next_level.name.title()} ({why})"):
                return False
        return True

    def take(course: Course, visiting: frozenset[str], reason: str) -> bool:
        """Add a course after its prerequisites are met."""
        if course.course_id in (c.course_id for c in path):
            return True                                   # already in the plan
        if course.course_id in visiting:
            warnings.append(f"Prerequisite loop at {course.course_id}")
            return False
        visiting = visiting | {course.course_id}
        for skill in sorted(course.prerequisites):
            if not climb(skill, Level.BEGINNER, visiting, f"needed for {course.course_id}"):
                return False
        path.append(PlannedCourse(course_id=course.course_id, course_name=course.course_name,
                                  duration_weeks=course.duration_weeks, reason=reason))
        for skill in course.skills_taught:                # learning raises every skill taught
            current[skill] = max(current.get(skill, Level.NONE), course.level)
        return True

    for gap in gaps:
        if not climb(gap.skill, gap.required, frozenset(), f"{gap.priority} for {family}"):
            uncovered.append(gap)

    return PlanResult(courses=path, uncovered=uncovered, warnings=sorted(set(warnings)))


def plan_problems(course_ids: list[str], levels: dict[str, Level], catalog: Catalog) -> list[str]:
    """Replay a plan in order and report anything wrong. Empty list = valid.

    Used to test the planner now, and to check agent output in M4.
    """
    problems = []
    current = dict(levels)
    for course_id in course_ids:
        course = catalog.course(course_id)
        if course is None:
            problems.append(f"{course_id} is not in the catalog")
            continue
        for skill in course.prerequisites:
            if current.get(skill, Level.NONE) < Level.BEGINNER:
                problems.append(f"{course_id} starts before {skill} is learned")
        for skill in course.skills_taught:
            current[skill] = max(current.get(skill, Level.NONE), course.level)
    return problems
