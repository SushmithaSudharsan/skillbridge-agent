"""M7: follow-up chat after the analysis. A supervisor agent with tools.

The analysis (fixed steps) already produced the facts. The chat agent answers the
student's questions about them by calling tools; it never sees raw files, only the
masked text and the computed comparison.

Tools:
  list_gaps, explain_gap, learning_plan, project_ideas   read the analysis
  prioritise        hands off to the M6 debate (a sub-agent), once, then cached
  record_answer     saves what the student confirms ("yes, I used Docker in my project")
  honest_wording    resume wording for a skill, ONLY after record_answer confirmed it,
                    and never above what the student said they did

Memory lives only in this object. end() (or leaving a `with` block) wipes it.
Nothing is written to disk or a database.
"""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from app.core import prompts
from app.services.agent import run_loop
from app.services.analysis import Analysis, Prepared
from app.services.debate import debate
from app.services.guardrails import quote_is_in
from app.services.learning import LEVEL_NAMES
from app.services.llm import LLM
from app.services.agent import Tool

How = Literal["tried", "used_in_project", "used_at_work"]
SUPPORTED_LEVEL = {"tried": "Beginner", "used_in_project": "Beginner", "used_at_work": "Intermediate"}
WORDING_HINT = {
    "tried": "Mention it under coursework or learning, e.g. 'Explored X through ...'. Not in experience.",
    "used_in_project": "Name it inside the project line where it was used, e.g. 'Used X to ...'.",
    "used_at_work": "Name it inside the job line, with what it did, e.g. 'Containerised ... with X'.",
}

SYSTEM_PROMPT = prompts.text("chat_system")


@dataclass
class ChatContext:
    prep: Prepared
    analysis: Analysis
    llm: LLM
    last_user: str = ""
    confirmed: dict[str, dict] = field(default_factory=dict)    # skill -> what the student said
    focus: dict | None = None


class NoArgs(BaseModel):
    pass


class SkillArg(BaseModel):
    skill: str = Field(description="A skill name from list_gaps")


class AnswerArgs(BaseModel):
    skill: str
    used: bool = Field(description="Did the student say they used it?")
    how: How = Field(description="tried = tutorial/course only; used_in_project; used_at_work")
    student_words: str = Field(description="The student's exact words from their last message")


class WordingArgs(BaseModel):
    skill: str
    resume_line: str = Field(description="Exact line from the resume where the skill was used")


def _row(ctx: ChatContext, skill: str):
    return next((m for m in ctx.prep.matches if m.skill.casefold() == skill.strip().casefold()), None)


def _list_gaps(ctx, _):
    return {"gaps": [{"skill": m.skill, "status": m.status, "current": LEVEL_NAMES[m.current],
                      "needed": LEVEL_NAMES[m.needed], "must_have": m.must_have} for m in ctx.prep.gaps],
            "strengths": [m.skill for m in ctx.analysis.strengths]}


def _explain(ctx, args: SkillArg):
    m = _row(ctx, args.skill)
    if m is None:
        return {"error": f"'{args.skill}' is not in this job's requirements. Call list_gaps."}
    steps = [{"course_area": s.text, "topics": s.topics} for s in ctx.prep.learning if s.skill == m.skill]
    return {"skill": m.skill, "status": m.status, "current": LEVEL_NAMES[m.current],
            "needed": LEVEL_NAMES[m.needed], "job_says": m.job_quote,
            "resume_shows": m.resume_quote or "nothing yet", "learning": steps}


def _plan(ctx, _):
    return {"steps": [{"course_area": s.text, "topics": s.topics} for s in ctx.prep.learning]}


def _projects(ctx, _):
    return {"projects": [p.model_dump() for p in ctx.analysis.projects]}


def _prioritise(ctx, _):
    if ctx.focus is None:                              # run the debate sub-agent once
        result = debate(ctx.llm, ctx.prep)
        ctx.focus = ({"focus": result.focus, "why": result.reason} if result else
                     {"focus": [m.skill for m in ctx.prep.gaps],
                      "why": "There are only a few gaps; work through them in this order."})
    return ctx.focus


def _record(ctx, args: AnswerArgs):
    m = _row(ctx, args.skill)
    if m is None:
        return {"error": f"'{args.skill}' is not in this job's requirements."}
    if not quote_is_in(args.student_words, ctx.last_user):
        return {"error": "student_words must be copied from the student's last message. Ask them."}
    ctx.confirmed[m.skill] = {"used": args.used, "how": args.how, "said": args.student_words}
    return {"saved_for_this_session": m.skill, "used": args.used}


def _wording(ctx, args: WordingArgs):
    m = _row(ctx, args.skill)
    if m is None:
        return {"error": f"'{args.skill}' is not in this job's requirements."}
    said = ctx.confirmed.get(m.skill)
    if not said:
        return {"error": f"First ask whether and how the student used {m.skill}, then call record_answer."}
    if not said["used"]:
        return {"advice": f"Don't add {m.skill}: the student said they haven't used it. Point them "
                          f"to the learning plan and a project instead."}
    if not quote_is_in(args.resume_line, ctx.prep.safe_resume):
        return {"error": "resume_line must be copied exactly from the resume."}
    return {"skill": m.skill, "resume_line": args.resume_line, "student_said": said["said"],
            "claim_at_most": SUPPORTED_LEVEL[said["how"]], "how_to_word_it": WORDING_HINT[said["how"]],
            "rules": "Describe only what the student said. No new numbers, tools or results."}


CHAT_TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("list_gaps", "The student's gaps and strengths for this job.", NoArgs, _list_gaps),
    Tool("explain_gap", "Why one skill is a gap: what the job says, what the resume shows, "
                        "and the course areas that close it.", SkillArg, _explain),
    Tool("learning_plan", "The full learning plan in order, as course areas with topics.", NoArgs, _plan),
    Tool("project_ideas", "Portfolio projects that show the missing skills.", NoArgs, _projects),
    Tool("prioritise", "What to work on first, and why. Asks a panel of advisors (slow).", NoArgs, _prioritise),
    Tool("record_answer", "Save what the student said about using a skill, in their own words.",
         AnswerArgs, _record),
    Tool("honest_wording", "Rules for adding a confirmed skill to a resume line. Call record_answer first.",
         WordingArgs, _wording),
]}


class ChatSession:
    """One student's follow-up chat. Use as `with ChatSession(...) as chat:`."""

    def __init__(self, llm: LLM, prep: Prepared, analysis: Analysis, max_steps: int = 6):
        self.llm, self.max_steps = llm, max_steps
        self.context = ChatContext(prep=prep, analysis=analysis, llm=llm)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT.format(comparison=prep.comparison())}]
        self.steps: list = []

    def ask(self, text: str) -> str:
        if self.context is None:
            raise RuntimeError("This chat has ended; its memory was wiped.")
        self.context.last_user = text
        self.messages.append({"role": "user", "content": text})
        answer, steps, _ = run_loop(self.llm, self.messages, CHAT_TOOLS, self.context, self.max_steps)
        self.steps += steps
        return answer

    def end(self) -> None:
        """Forget everything: messages, confirmed answers, the masked resume."""
        self.messages.clear()
        self.steps.clear()
        self.context = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.end()
