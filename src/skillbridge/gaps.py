"""Compare a student's skills with what a role needs."""

from skillbridge.models import Level, RoleProfile, SkillGap, StudentSkill

PRIORITY_ORDER = {"required": 0, "important": 1}


def student_levels(skills: list[StudentSkill]) -> dict[str, Level]:
    """One level per skill. If a resume mentions a skill twice, keep the higher level."""
    levels: dict[str, Level] = {}
    for item in skills:
        levels[item.skill] = max(levels.get(item.skill, Level.NONE), item.level)
    return levels


def find_gaps(levels: dict[str, Level], profile: RoleProfile) -> list[SkillGap]:
    """Every skill where the student is below the level the role needs.

    A skill missing from `levels` counts as Level.NONE: experience in other
    skills never fills a gap on its own.
    """
    gaps = []
    for req in profile.requirements:
        current = levels.get(req.skill, Level.NONE)
        if current < req.level:
            gaps.append(SkillGap(skill=req.skill, current=current,
                                 required=req.level, priority=req.priority))
    gaps.sort(key=lambda g: (PRIORITY_ORDER[g.priority], -g.size, g.skill))
    return gaps
