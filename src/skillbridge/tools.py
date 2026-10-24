"""The tools the agent can call. Plain Python on top of M1: the LLM decides, Python answers.

Every tool:
  - has a clear description (the only thing the LLM knows about it)
  - checks its own inputs and returns errors as data the LLM can act on
  - reads the student's skills from the context, never from the LLM's arguments
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from skillbridge.catalog import Catalog
from skillbridge.experience import decide_seniority
from skillbridge.gaps import find_gaps, student_levels
from skillbridge.market import MarketProfile, requirements_for
from skillbridge.models import Level, RoleProfile, StudentSkill
from skillbridge.planner import make_plan

SeniorityArg = Literal["fresher", "mid", "senior"]


@dataclass
class AgentContext:
    """Facts the tools read. The LLM can't see or change these directly."""

    catalog: Catalog
    student_skills: list[StudentSkill]
    market: list[MarketProfile] = field(default_factory=list)

    @property
    def levels(self) -> dict[str, Level]:
        return student_levels(self.student_skills)


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[[AgentContext, Any], dict]

    def openai_spec(self) -> dict:
        """The tool description in the common JSON format LLM providers accept."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": _clean(self.args_model.model_json_schema())}}


def _clean(schema):
    """Drop keys some providers reject (titles, defaults), keeping the shape of the schema.

    Careful: inside "properties" the keys are argument NAMES (an argument may be called
    "title"), so those are kept and only their schemas are cleaned.
    """
    if isinstance(schema, list):
        return [_clean(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    cleaned = {}
    for key, value in schema.items():
        if key == "properties":
            cleaned[key] = {name: _clean(prop) for name, prop in value.items()}
        elif key not in ("title", "default"):
            cleaned[key] = _clean(value)
    return cleaned


def call_tool(tools: dict[str, Tool], context: AgentContext, name: str, args: dict) -> dict:
    """Run one tool safely. Bad names and bad arguments come back as messages, not crashes."""
    tool = tools.get(name)
    if tool is None:
        return {"error": f"Unknown tool '{name}'. Available tools: {sorted(tools)}"}
    try:
        parsed = tool.args_model.model_validate(args)
    except ValidationError as error:
        return {"error": f"Invalid arguments for {name}: {error.errors(include_url=False)}"}
    try:
        return tool.run(context, parsed)
    except Exception as error:                      # a bug in a tool must not kill the agent
        return {"error": f"{name} failed: {error}"}


# ---------- Helpers ----------

def _check_family(context: AgentContext, family: str) -> dict | None:
    names = [f.name for f in context.catalog.families]
    if family not in names:
        return {"error": f"Unknown family '{family}'. Use match_job_title first. Families: {names}"}
    return None


def _requirements(context: AgentContext, family: str, seniority: str):
    reqs, source = requirements_for(context.catalog, family, seniority, context.market)
    return RoleProfile(family=family, seniority=seniority, requirements=reqs), source


# ---------- 1. match_job_title ----------

class MatchTitleArgs(BaseModel):
    title: str = Field(description="The job title the student wants, in their own words")
    jd_text: str = Field(default="", description="The job description text, if the student gave one")


def _match_title(context: AgentContext, args: MatchTitleArgs) -> dict:
    match = context.catalog.parse_title(args.title)
    if match is None:
        return {"supported": False,
                "message": "SkillBridge supports AI/ML and data roles only.",
                "supported_families": [f.name for f in context.catalog.families]}
    decision = decide_seniority(match.seniority, args.jd_text)
    return {"supported": True, "family": match.family, "seniority": decision.seniority,
            "seniority_source": decision.source, "note": decision.note}


# ---------- 2. get_student_skills ----------

class NoArgs(BaseModel):
    pass


def _student_skills(context: AgentContext, args: NoArgs) -> dict:
    return {"skills": [{"skill": s.skill, "level": s.level.name.title(),
                        "confidence": s.confidence, "evidence": s.evidence}
                       for s in sorted(context.student_skills, key=lambda s: s.skill)]}


# ---------- 3. find_skill_gaps ----------

class RoleArgs(BaseModel):
    family: str = Field(description="Role family exactly as returned by match_job_title")
    seniority: SeniorityArg


def _find_gaps(context: AgentContext, args: RoleArgs) -> dict:
    if (error := _check_family(context, args.family)):
        return error
    profile, source = _requirements(context, args.family, args.seniority)
    levels = context.levels
    gaps = find_gaps(levels, profile)
    shares = {r.skill: r.share for r in profile.requirements}
    met = sorted(r.skill for r in profile.requirements if levels.get(r.skill, Level.NONE) >= r.level)
    return {"requirements_source": source,
            "gaps": [{"skill": g.skill, "current": g.current.name.title(),
                      "required": g.required.name.title(), "priority": g.priority,
                      "steps_to_climb": g.size, "posting_share": shares.get(g.skill)}
                     for g in gaps],
            "already_meets": met}


# ---------- 4. plan_learning_path ----------

class PlanArgs(RoleArgs):
    focus_skills: list[str] = Field(
        default_factory=list,
        description="Only plan these gap skills (e.g. if the student has little time). Empty = all")


def _plan(context: AgentContext, args: PlanArgs) -> dict:
    if (error := _check_family(context, args.family)):
        return error
    profile, source = _requirements(context, args.family, args.seniority)
    levels = context.levels
    gaps = find_gaps(levels, profile)
    if args.focus_skills:
        wanted = {s.casefold() for s in args.focus_skills}
        gaps = [g for g in gaps if g.skill.casefold() in wanted]
    plan = make_plan(gaps, levels, context.catalog, args.family)
    return {"requirements_source": source,
            "courses": [{"course_id": c.course_id, "name": c.course_name,
                         "weeks": c.duration_weeks, "reason": c.reason} for c in plan.courses],
            "total_weeks": plan.total_weeks,
            "uncovered_gaps": [g.skill for g in plan.uncovered],
            "warnings": plan.warnings}


# ---------- 5. search_courses ----------

class SearchArgs(BaseModel):
    skill: str = Field(description="Skill to find courses for, e.g. 'RAG'")
    level: Literal["Any", "Beginner", "Intermediate", "Advanced"] = "Any"


def _search(context: AgentContext, args: SearchArgs) -> dict:
    found = [c for c in context.catalog.courses
             if args.skill.casefold() in (s.casefold() for s in c.skills_taught)
             and args.level in ("Any", c.difficulty)]
    found.sort(key=lambda c: (c.level, c.duration_weeks, c.course_id))
    if not found:
        return {"courses": [], "message": f"No courses teach '{args.skill}'. "
                                         f"Known skills: {context.catalog.skills}"}
    return {"courses": [{"course_id": c.course_id, "name": c.course_name, "level": c.difficulty,
                         "weeks": c.duration_weeks, "prerequisites": c.prerequisites}
                        for c in found]}


# ---------- 6. recommend_projects ----------

class ProjectArgs(RoleArgs):
    max_results: int = Field(default=3, ge=1, le=5)


def _projects(context: AgentContext, args: ProjectArgs) -> dict:
    if (error := _check_family(context, args.family)):
        return error
    profile, _ = _requirements(context, args.family, args.seniority)
    gap_skills = {g.skill for g in find_gaps(context.levels, profile)}
    known = set(context.levels)

    def score(p):
        covers = len(gap_skills & set(p.skills))
        uses_known = len(known & set(p.skills))
        for_role = args.family in p.careers
        return (-for_role, -covers, -uses_known, p.project_id)

    chosen = sorted(context.catalog.projects, key=score)[: args.max_results]
    return {"projects": [{"project_id": p.project_id, "name": p.project_name,
                          "description": p.description, "difficulty": p.difficulty,
                          "shows_gap_skills": sorted(gap_skills & set(p.skills))}
                         for p in chosen]}


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("match_job_title",
         "Map a job title (and optional job description) to a supported role family and "
         "seniority. Call this first whenever the student names a job.",
         MatchTitleArgs, _match_title),
    Tool("get_student_skills",
         "List the student's skills with level, confidence and the resume evidence for each.",
         NoArgs, _student_skills),
    Tool("find_skill_gaps",
         "Compare the student's skills with a role's requirements. Returns gaps (required first) "
         "with current and required level, plus the skills already met and where the "
         "requirements came from.",
         RoleArgs, _find_gaps),
    Tool("plan_learning_path",
         "Build an ordered course plan that closes the gaps, prerequisites first. Returns course "
         "IDs with reasons, total weeks, and any gaps no course covers.",
         PlanArgs, _plan),
    Tool("search_courses",
         "Look up catalog courses for one skill, optionally at one level.",
         SearchArgs, _search),
    Tool("recommend_projects",
         "Suggest portfolio projects for the role that show the student's gap skills.",
         ProjectArgs, _projects),
]}
