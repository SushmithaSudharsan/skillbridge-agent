"""Read a resume with an LLM, then keep only skills the resume really proves."""

from typing import Literal

from pydantic import BaseModel, Field

from skillbridge.guardrails import mask_pii, quote_is_in, wrap_document
from skillbridge.llm import LLM
from skillbridge.matching import SkillMatcher
from skillbridge.models import Level, StudentSkill

# ---------- What we ask the LLM to fill in ----------

class ResumeSkillDraft(BaseModel):
    skill: str = Field(description="The skill name, using a name from the vocabulary when one fits")
    level: int = Field(ge=1, le=3, description=(
        "1 = listed, 'basic', or only studied; 2 = used in a job or in two or more projects; "
        "3 = led, optimised, taught, or used deeply at work. Total years of experience never "
        "raises a single skill."))
    source: Literal["job", "project", "certification", "course", "self_study", "listed"]
    evidence: str = Field(description="Exact words copied from the resume that prove this skill")


class ResumeDraft(BaseModel):
    skills: list[ResumeSkillDraft]


PROMPT = """Extract the technical and professional skills this resume proves.
Rules:
- Only skills the resume actually shows. Never add skills it does not mention.
- 'evidence' must be copied word for word from the resume.
- Use these skill names when they fit: {vocabulary}

{resume}"""


# ---------- What we return ----------

class Rejected(BaseModel):
    skill: str
    reason: str


class ResumeProfile(BaseModel):
    skills: list[StudentSkill]            # usable for gap analysis
    other_skills: list[StudentSkill]      # proven, but not in our vocabulary yet
    rejected: list[Rejected]              # failed the no-invention check


def read_resume(llm: LLM, resume_text: str, matcher: SkillMatcher,
                vocabulary: list[str]) -> ResumeProfile:
    safe_text = mask_pii(resume_text)
    prompt = PROMPT.format(vocabulary=", ".join(vocabulary),
                           resume=wrap_document("resume", safe_text))
    draft: ResumeDraft = llm.extract(ResumeDraft, prompt)
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
        skill = StudentSkill(skill=canonical or item.skill, level=Level(item.level),
                             source=item.source, evidence=item.evidence)
        (skills if canonical else other).append(skill)
    skills.sort(key=lambda s: s.skill)
    other.sort(key=lambda s: s.skill)
    return ResumeProfile(skills=skills, other_skills=other, rejected=rejected)
