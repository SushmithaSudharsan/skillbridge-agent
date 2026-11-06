"""Read a resume with an LLM, then keep only skills the resume really proves."""

from typing import Literal

from pydantic import BaseModel, Field

from app.core import prompts
from app.services.jd import clean_name
from app.services.guardrails import mask_pii, quote_is_in, wrap_document
from app.services.llm import LLM
from app.services.matching import SkillMatcher
from app.services.models import Level, StudentSkill

# ---------- What we ask the LLM to fill in ----------

class ResumeSkillDraft(BaseModel):
    skill: str = Field(description="The skill name, using a name from the vocabulary when one fits")
    level: int = Field(ge=1, le=3, description=(
        "1 = only listed, called 'basic', one course, or one small project; "
        "2 = used in a job or in two or more projects, OR the major/specialisation of a degree, "
        "OR two or more courses in it; "
        "3 = led, optimised, taught, or used deeply at work. Total years of experience never "
        "raises a single skill."))
    source: Literal["job", "project", "certification", "course", "self_study", "listed"] = Field(
        description="course = coursework or a degree; listed = only named in a skills list")
    evidence: str = Field(description="Exact words copied from the resume that prove this skill")


class ResumeDraft(BaseModel):
    skills: list[ResumeSkillDraft]


PROMPT = prompts.text("resume_read")


# The highest level each kind of evidence can prove. Decided in code, whatever the LLM says:
# a skills list can't prove more than Beginner, and study alone can't prove Advanced.
MAX_LEVEL = {"listed": Level.BEGINNER, "course": Level.INTERMEDIATE, "self_study": Level.INTERMEDIATE,
             "certification": Level.INTERMEDIATE, "project": Level.ADVANCED, "job": Level.ADVANCED}


# ---------- What we return ----------

class Rejected(BaseModel):
    skill: str
    reason: str


class ResumeProfile(BaseModel):
    skills: list[StudentSkill]            # usable for gap analysis
    other_skills: list[StudentSkill] = []   # filled by the analysis: proven, but not asked for by this job
    rejected: list[Rejected]              # failed the no-invention check


def read_resume(llm: LLM, resume_text: str, matcher: SkillMatcher,
                vocabulary: list[str], private_terms: list[str] | None = None,
                requirements: list[str] | None = None) -> ResumeProfile:
    safe_text = mask_pii(resume_text, private_terms)
    prompt = PROMPT.format(vocabulary=", ".join(vocabulary),
                           requirements="\n".join(f"- {r}" for r in requirements or []) or "- (none given)",
                           resume=wrap_document("resume", safe_text))
    draft: ResumeDraft = llm.extract(ResumeDraft, prompt, system=prompts.READER)
    return check_draft(draft, safe_text, matcher)


def check_draft(draft: "ResumeDraft", resume_text: str, matcher: SkillMatcher) -> ResumeProfile:
    """Python verifies what the LLM claimed. This is where invented skills get caught."""
    skills, other, rejected = [], [], []
    for item in draft.skills:
        if not quote_is_in(item.evidence, resume_text):
            rejected.append(Rejected(skill=item.skill,
                                     reason=f"evidence not found in resume: '{item.evidence}'"))
            continue
        canonical = matcher.match(item.skill)
        level = min(Level(item.level), MAX_LEVEL[item.source])
        skills.append(StudentSkill(skill=canonical or clean_name(item.skill), level=level,
                                   source=item.source, evidence=item.evidence))
    skills.sort(key=lambda s: s.skill)
    other.sort(key=lambda s: s.skill)
    return ResumeProfile(skills=skills, other_skills=other, rejected=rejected)
