"""Reading a job description: the LLM lists requirements, Python checks each quote."""

from typing import Literal

from pydantic import BaseModel, Field

from app.services.guardrails import quote_is_in
from app.services.matching import SkillMatcher
from app.services.models import Level


# ---------- What we ask the LLM to fill in for one posting ----------

Kind = Literal["skill", "task", "soft"]


class PostingSkillDraft(BaseModel):
    skill: str = Field(description="Requirement name, using a name from the vocabulary when one fits")
    kind: Kind = Field(default="skill", description="skill = tool/technology/method; task = work or "
                                                    "domain experience; soft = way of working")
    level: int = Field(ge=1, le=3, description=(
        "1 = 'familiarity', 'exposure', 'basic', 'nice to have'; 2 = 'experience with', "
        "'hands-on', plain mention; 3 = 'strong', 'expert', 'deep', 'lead'"))
    must_have: bool = Field(description="True if listed as required, false if a plus or preferred")
    evidence: str = Field(description="Exact words copied from the posting")


class PostingDraft(BaseModel):
    skills: list[PostingSkillDraft]


class PostingSkill(BaseModel):
    skill: str
    level: Level
    must_have: bool
    evidence: str
    in_catalog: bool
    kind: Kind = "skill"


class PostingExtraction(BaseModel):
    posting_id: str
    text_hash: str
    skills: list[PostingSkill]
    rejected: list[str] = Field(default_factory=list)


def check_posting_draft(draft: PostingDraft, posting_id: str, text: str, text_hash: str,
                        matcher: SkillMatcher) -> PostingExtraction:
    skills, rejected = [], []
    for item in draft.skills:
        if not quote_is_in(item.evidence, text):
            rejected.append(f"{item.skill}: evidence not in posting")
            continue
        canonical = matcher.match(item.skill)
        skills.append(PostingSkill(skill=canonical or clean_name(item.skill), level=Level(item.level),
                                   must_have=item.must_have, evidence=item.evidence,
                                   in_catalog=canonical is not None, kind=item.kind))
    return PostingExtraction(posting_id=posting_id, text_hash=text_hash,
                             skills=skills, rejected=rejected)


def clean_name(name: str) -> str:
    """A requirement name outside the vocabulary: trimmed, first letter capitalised, rest kept
    (so 'PySpark' and 'A/B testing' survive)."""
    name = " ".join(name.strip(" .,;:-").split())
    return name[:1].upper() + name[1:] if name else name
