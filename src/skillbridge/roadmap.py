"""The final answer as a typed object, and code checks on it."""

from pydantic import BaseModel, Field

from skillbridge.planner import plan_problems
from skillbridge.tools import AgentContext


class RoadmapCourse(BaseModel):
    course_id: str
    why: str = Field(description="One sentence: which gap and level this closes")


class Roadmap(BaseModel):
    target_family: str
    seniority: str
    requirements_source: str
    summary: str = Field(description="Two or three sentences on where the student stands")
    courses: list[RoadmapCourse] = Field(description="In the order returned by plan_learning_path")
    project_ids: list[str]
    uncovered_gaps: list[str]
    next_two_weeks: list[str] = Field(description="Concrete first steps")


def roadmap_problems(roadmap: Roadmap, context: AgentContext) -> list[str]:
    """Anything invented or out of order. M4 will send these back to the agent to fix."""
    problems = plan_problems([c.course_id for c in roadmap.courses], context.levels,
                             context.catalog)
    project_ids = {p.project_id for p in context.catalog.projects}
    for pid in roadmap.project_ids:
        if pid not in project_ids:
            problems.append(f"Project {pid} is not in the catalog")
    return problems
