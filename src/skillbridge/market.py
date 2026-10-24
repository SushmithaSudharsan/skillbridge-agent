"""Requirements from real job postings: the LLM reads each posting, Python counts."""

import hashlib
from collections import defaultdict
from pathlib import Path
from statistics import median_low

from pydantic import BaseModel, Field

from skillbridge.catalog import DEFAULT_DATA_DIR, Catalog
from skillbridge.experience import decide_seniority
from skillbridge.guardrails import mask_pii, quote_is_in, wrap_document
from skillbridge.llm import LLM
from skillbridge.matching import SkillMatcher
from skillbridge.models import Level, Requirement

REQUIRED_SHARE = 0.40      # in at least 40% of postings -> required
IMPORTANT_SHARE = 0.15     # in at least 15% -> important
MIN_POSTINGS = 5           # fewer than this -> not enough evidence; use the curated profile


# ---------- Postings on disk ----------

class Posting(BaseModel):
    id: str
    title: str
    text: str
    label: str | None = None      # your own seniority label, used only for evaluation


def load_postings(folder: Path | None = None) -> list[Posting]:
    """Read every .txt file in data/sample_jds. Header lines look like '# title: AI Engineer'."""
    folder = folder or DEFAULT_DATA_DIR / "sample_jds"
    postings = []
    for path in sorted(folder.glob("*.txt")):
        header, body = {}, []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") and ":" in line and not body:
                key, value = line[1:].split(":", 1)
                header[key.strip().lower()] = value.strip()
            else:
                body.append(line)
        if "title" not in header:
            raise ValueError(f"{path.name}: add a first line like '# title: AI Engineer'")
        postings.append(Posting(id=path.stem, title=header["title"], label=header.get("label"),
                                text="\n".join(body).strip()))
    return postings


# ---------- What we ask the LLM to fill in for one posting ----------

class PostingSkillDraft(BaseModel):
    skill: str = Field(description="Skill name, using a name from the vocabulary when one fits")
    level: int = Field(ge=1, le=3, description=(
        "1 = 'familiarity', 'exposure', 'basic', 'nice to have'; 2 = 'experience with', "
        "'hands-on', plain mention; 3 = 'strong', 'expert', 'deep', 'lead'"))
    must_have: bool = Field(description="True if listed as required, false if a plus or preferred")
    evidence: str = Field(description="Exact words copied from the posting")


class PostingDraft(BaseModel):
    skills: list[PostingSkillDraft]


PROMPT = """List every skill this job posting asks for.
Rules:
- Only skills the posting mentions. 'evidence' must be copied word for word.
- Use these skill names when they fit: {vocabulary}

{posting}"""


class PostingSkill(BaseModel):
    skill: str
    level: Level
    must_have: bool
    evidence: str
    in_catalog: bool


class PostingExtraction(BaseModel):
    posting_id: str
    text_hash: str
    skills: list[PostingSkill]
    rejected: list[str] = Field(default_factory=list)


def extract_posting(llm: LLM, posting: Posting, matcher: SkillMatcher,
                    vocabulary: list[str], cache_dir: Path | None = None) -> PostingExtraction:
    """Ask the LLM once per posting; reuse the saved answer if the posting has not changed."""
    text = mask_pii(posting.text)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    cache_file = (cache_dir or DEFAULT_DATA_DIR / "cache" / "postings") / f"{posting.id}.json"
    if cache_file.exists():
        cached = PostingExtraction.model_validate_json(cache_file.read_text(encoding="utf-8"))
        if cached.text_hash == text_hash:
            return cached

    prompt = PROMPT.format(vocabulary=", ".join(vocabulary), posting=wrap_document("posting", text))
    draft: PostingDraft = llm.extract(PostingDraft, prompt)
    result = check_posting_draft(draft, posting.id, text, text_hash, matcher)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def check_posting_draft(draft: PostingDraft, posting_id: str, text: str, text_hash: str,
                        matcher: SkillMatcher) -> PostingExtraction:
    skills, rejected = [], []
    for item in draft.skills:
        if not quote_is_in(item.evidence, text):
            rejected.append(f"{item.skill}: evidence not in posting")
            continue
        canonical = matcher.match(item.skill)
        skills.append(PostingSkill(skill=canonical or item.skill, level=Level(item.level),
                                   must_have=item.must_have, evidence=item.evidence,
                                   in_catalog=canonical is not None))
    return PostingExtraction(posting_id=posting_id, text_hash=text_hash,
                             skills=skills, rejected=rejected)


# ---------- Counting across postings ----------

class MarketProfile(BaseModel):
    family: str
    seniority: str
    posting_ids: list[str]
    requirements: list[Requirement]
    not_in_catalog: dict[str, float]     # skills employers ask for that we have no course for yet

    @property
    def n_postings(self) -> int:
        return len(self.posting_ids)


def aggregate(family: str, seniority: str, extractions: list[PostingExtraction]) -> MarketProfile:
    """Share = postings asking for a skill / all postings. Level = middle level asked for."""
    n = len(extractions)
    asked: dict[str, list[Level]] = defaultdict(list)
    for extraction in extractions:
        per_posting: dict[str, Level] = {}
        for s in extraction.skills:
            per_posting[s.skill] = max(per_posting.get(s.skill, Level.NONE), s.level)
        for skill, level in per_posting.items():
            asked[skill].append(level)

    catalog_skills = {s.skill for e in extractions for s in e.skills if s.in_catalog}
    requirements, outside = [], {}
    for skill, levels in sorted(asked.items()):
        share = round(len(levels) / n, 2)
        if share < IMPORTANT_SHARE:
            continue
        if skill not in catalog_skills:
            outside[skill] = share
            continue
        requirements.append(Requirement(
            skill=skill, level=Level(median_low(levels)), share=share,
            priority="required" if share >= REQUIRED_SHARE else "important"))
    return MarketProfile(family=family, seniority=seniority,
                         posting_ids=sorted(e.posting_id for e in extractions),
                         requirements=requirements, not_in_catalog=outside)


def build_market(llm: LLM, catalog: Catalog, matcher: SkillMatcher,
                 postings: list[Posting], cache_dir: Path | None = None) -> list[MarketProfile]:
    """Group postings by role family and seniority, extract each, and count."""
    groups: dict[tuple[str, str], list[PostingExtraction]] = defaultdict(list)
    for posting in postings:
        match = catalog.parse_title(posting.title)
        if match is None:
            continue
        decision = decide_seniority(match.seniority, posting.text)
        if decision.seniority is None:
            continue
        groups[(match.family, decision.seniority)].append(
            extract_posting(llm, posting, matcher, catalog.skills, cache_dir))
    return [aggregate(family, seniority, ex) for (family, seniority), ex in sorted(groups.items())]


def requirements_for(catalog: Catalog, family: str, seniority: str,
                     market: list[MarketProfile]) -> tuple[list[Requirement], str]:
    """Market requirements when there is enough evidence, otherwise the curated profile."""
    for profile in market:
        if profile.family == family and profile.seniority == seniority:
            if profile.n_postings >= MIN_POSTINGS:
                return profile.requirements, f"job market ({profile.n_postings} postings)"
            source_note = (f"curated profile (only {profile.n_postings} posting(s) collected; "
                           f"need {MIN_POSTINGS})")
            break
    else:
        source_note = "curated profile (no postings collected yet)"
    curated = catalog.profile(family, seniority)
    return (curated.requirements if curated else []), source_note
