"""The main SkillBridge analysis: one resume against one job description.

Fixed steps, 3 LLM calls, every LLM claim checked by Python:
  1. Read the JD      (LLM)  -> requirements, each with a quote from the JD
  2. Read the resume  (LLM)  -> skills, each with a quote from the resume
  3. Compare          (code) -> meets / below / missing, per requirement
  4. Learn            (code) -> course areas and topics, prerequisites first
  5. Advise           (LLM)  -> summary, project ideas, resume questions
  6. Check advice     (code) -> drop anything that quotes text that isn't there
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from app.core import prompts
from app.services.catalog import Catalog
from app.services.experience import ExperienceRules, decide_seniority, experience_rules, years_required
from app.services.blockers import Blocker, find_blockers
from app.services.fit import Fit, assess_fit
from app.services.guardrails import mask_pii, quote_is_in, rewrite_problem, wrap_document
from app.services.learning import LEVEL_NAMES, LearningStep, SkillGuide, learning_steps
from app.services.llm import LLM
from app.services.jd import PostingDraft, check_posting_draft
from app.services.related import peers as related_peers
from app.services.matching import SkillMatcher
from app.services.models import PRIORITY_ORDER, Level, SkillGap, StudentSkill
from app.services.resume import ResumeProfile, read_resume

# ---------- Comparing, requirement by requirement ----------

Visibility = Literal["shown", "listed", "implied", "answer", "none"]

class SkillMatch(BaseModel):
    skill: str
    needed: Level
    must_have: bool
    job_quote: str
    current: Level
    resume_quote: str | None
    in_guide: bool
    kind: str = "skill"                         # skill / task / soft
    visibility: Visibility = "none"             # how the resume shows it
    related: str | None = None                  # a close tool the resume does show (never counts as meeting it)
    related_level: Level = Level.NONE
    related_quote: str | None = None

    @property
    def is_soft(self) -> bool:
        return self.kind == "soft"

    @property
    def status(self) -> Literal["meets", "below", "missing"]:
        if self.current >= self.needed:
            return "meets"
        return "below" if self.current > Level.NONE else "missing"


def compare(job_skills, student_skills: list[StudentSkill]) -> list[SkillMatch]:
    """One row per JD requirement. A skill counts only with resume evidence."""
    best: dict[str, StudentSkill] = {}
    for s in student_skills:
        if s.skill not in best or (s.level, s.source != "listed") > (best[s.skill].level, best[s.skill].source != "listed"):
            best[s.skill] = s      # higher level wins; on a tie, real use beats a skills-list mention
    merged: dict[str, SkillMatch] = {}
    for req in job_skills:
        have = best.get(req.skill)
        row = SkillMatch(skill=req.skill, needed=req.level, must_have=req.must_have,
                         job_quote=req.evidence, in_guide=req.in_catalog, kind=req.kind,
                         visibility=("none" if not have else "listed" if have.source == "listed"
                                     else "implied" if have.source == "implied" else "shown"),
                         current=have.level if have else Level.NONE,
                         resume_quote=(f"Implied by your {have.via} work: {have.evidence}" if have and have.via
                                       else have.evidence if have else None))
        if row.status != "meets" and row.kind == "skill":
            close = [best[p] for p in related_peers(req.skill) if p in best and best[p].level > Level.NONE
                     and best[p].source != "listed"]
            if close:
                top = max(close, key=lambda c: c.level)
                row = row.model_copy(update={"related": top.skill, "related_level": top.level,
                                             "related_quote": top.evidence})
        old = merged.get(req.skill)
        if old is None or (row.needed, row.must_have) > (old.needed, old.must_have):
            merged[req.skill] = row
    return sort_matches(list(merged.values()))


def sort_matches(rows: list[SkillMatch]) -> list[SkillMatch]:
    """Gaps first, must-haves first, biggest gaps first."""
    return sorted(rows, key=lambda r: (r.status == "meets", not r.must_have, -(r.needed - r.current), r.skill))


# ---------- Resume checks that need no LLM ----------

SECTION_WORDS = {"experience": ["experience", "employment", "work history"],
                 "education": ["education"], "skills": ["skills"], "projects": ["projects"]}


def resume_checks(original_text: str, profile: ResumeProfile, matches: list[SkillMatch]) -> list[dict]:
    """Basic ATS readability, and 'listed but never shown' (the opposite of keyword stuffing)."""
    lower = original_text.casefold()
    checks = []
    found = [name for name, words in SECTION_WORDS.items() if any(w in lower for w in words)]
    missing = [name for name in SECTION_WORDS if name not in found]
    checks.append({"ok": not missing, "area": "Sections",
                   "text": "Standard sections found: " + ", ".join(found) if not missing
                   else "Add standard section headings: " + ", ".join(missing)})
    has_contact = "@" in original_text or bool(re.search(r"\d{10}", re.sub(r"\D", "", original_text)))
    checks.append({"ok": has_contact, "area": "Contact details",
                   "text": "Found." if has_contact else "No email or phone found. Recruiters need one."})
    words = len(original_text.split())
    checks.append({"ok": 250 <= words <= 900, "area": "Length",
                   "text": f"{words} words. " + ("Fits one to two pages." if 250 <= words <= 900
                                                  else "Aim for roughly 250-900 words.")})
    needed = {m.skill for m in matches}
    shown = {s.skill for s in profile.skills if s.source != "listed"}
    listed_only = sorted(s.skill for s in profile.skills
                         if s.source == "listed" and s.skill in needed and s.skill not in shown)
    if listed_only:
        checks.append({"ok": False, "area": "Listed but not shown", "skills": listed_only,
                       "text": "The job asks for " + ", ".join(listed_only) + ", which your resume only "
                               "lists. Show where you used them in a project or job; a bare list can "
                               "read as keyword stuffing."})
    return checks


# ---------- Advice from the LLM, checked by Python ----------

class ProjectIdea(BaseModel):
    title: str
    build: str = Field(description="One or two sentences: what the student builds")
    job_quote: str = Field(description="Exact words copied from the JD this project shows")
    skills: list[str] = Field(description="Skills from the JD requirements this project shows")


class ResumeQuestion(BaseModel):
    resume_line: str = Field(description="Exact words copied from the resume")
    issue: str = Field(description="What is weak or unclear about this line, in one sentence")
    question: str = Field(description="A question to the student. Never suggest adding something they haven't said.")


class ShowBetterDraft(BaseModel):
    skill: str = Field(description="A requirement the student already meets")
    resume_line: str = Field(description="Exact words copied from the resume")
    suggestion: str = Field(description="How to present these same facts for this job, in one or two sentences")


class LearningSuggestion(BaseModel):
    skill: str
    area: str = Field(description="A course area, e.g. 'Fundamentals of time-series forecasting'")
    topics: list[str] = Field(default_factory=list, description="Up to 4 key topics")


class AdviceDraft(BaseModel):
    summary: str = Field(description="Three sentences: overall fit (as computed), main strengths, main gaps")
    projects: list[ProjectIdea] = Field(description="Two or three portfolio projects")
    show_better: list[ShowBetterDraft] = Field(default_factory=list)
    learning: list[LearningSuggestion] = Field(default_factory=list)


class ShowBetter(BaseModel):
    """A requirement the student meets, but the resume doesn't make visible for this job."""
    skill: str
    why: Literal["listed", "implied", "context"]
    resume_line: str | None
    text: str


PROVIDER = re.compile(r"\b(coursera|udemy|edx|udacity|linkedin learning|pluralsight|datacamp|nptel|"
                      r"simplilearn|great learning|codecademy|skillshare|khan academy)\b"
                      r"|\b[A-Z]{2,4}\s?-?\d{3,4}\b|\b\d+\s*(?:weeks?|hours?|months?)\b", re.IGNORECASE)


ADVICE_PROMPT = prompts.text("advise")


class GapQuestion(BaseModel):
    skill: str
    issue: str
    question: str


class Change(BaseModel):
    """A skill whose level changed (up or down) because of the student's own answer."""
    skill: str
    before: Level
    after: Level
    said: str                 # the student's words (masked)
    source: str
    where: str                # what to do on the resume
    question: int = -1        # which question (index in Analysis.asked()) it came from


class Analysis(BaseModel):
    title: str
    seniority: str | None
    seniority_note: str
    summary: str
    matches: list[SkillMatch]
    learning: list[LearningStep]
    projects: list[ProjectIdea]
    questions: list[ResumeQuestion]
    checks: list[dict]
    other_skills: list[str]          # proven on the resume but not in our vocabulary
    dropped: list[str]               # LLM output removed by the checks (for transparency)
    fit: Fit | None = None                               # is this job a realistic target now?
    focus: list[str] = Field(default_factory=list)       # M6: what to work on first
    focus_reason: str = ""
    review: dict | None = None                           # M5: the judge's verdict
    show_better: list[ShowBetter] = Field(default_factory=list)
    gap_questions: list[GapQuestion] = Field(default_factory=list)
    blockers: list[Blocker] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)     # answers to questions + gap_questions, in order
    changes: list[Change] = Field(default_factory=list)  # what the answers changed
    notes: dict[int, dict] = Field(default_factory=dict) # per question: takeaway, rewrite, effects

    def answer_for(self, i: int) -> str:
        return self.answers[i].strip() if i < len(self.answers) else ""

    def asked(self) -> list[dict]:
        """Every question asked, in order, with the student's answer ('' if skipped)."""
        rows = [{"topic": f'"{q.resume_line}"', "issue": q.issue, "question": q.question} for q in self.questions]
        rows += [{"topic": f"About {g.skill}", "issue": g.issue, "question": g.question} for g in self.gap_questions]
        out = []
        for i, r in enumerate(rows):
            row = {**r, "answer": self.answer_for(i), "takeaway": "", "rewrite": "", "effects": []}
            row.update(self.notes.get(i, {}))
            out.append(row)
        return out

    @property
    def gaps(self) -> list[SkillMatch]:
        """Skills and experience to work on. Soft skills are never a gap: see `soft`."""
        return [m for m in self.matches if m.status != "meets" and not m.is_soft]

    @property
    def soft(self) -> dict[str, list[SkillMatch]]:
        """Ways of working the job names. 'shown' = a resume line proves it; 'interview' = nothing to
        fix on the resume, prepare a real example for the interview."""
        rows = [m for m in self.matches if m.is_soft]
        return {"shown": [m for m in rows if m.status == "meets"],
                "interview": [m for m in rows if m.status != "meets"]}

    @property
    def strengths(self) -> list[SkillMatch]:
        """Met, and the resume already shows it well for this job."""
        hidden = {b.skill for b in self.show_better}
        return [m for m in self.matches if m.status == "meets" and m.skill not in hidden]


class Prepared(BaseModel):
    """Everything from steps 1-4: facts that later steps (advice, judge, debate, chat) reuse."""

    title: str
    safe_jd: str
    safe_resume: str
    matches: list[SkillMatch]
    learning: list[LearningStep]
    checks: list[dict]
    other_skills: list[str]
    dropped: list[str]
    seniority: str | None
    seniority_note: str
    levels: dict[str, int] = Field(default_factory=dict)   # every skill read, JD or not (prerequisites)
    role_years: float | None = None                        # years the JD asks for
    your_years: float | None = None                        # the student's full-time work years
    internship_months: float = 0
    rules: ExperienceRules = Field(default_factory=ExperienceRules)   # what the JD counts as experience
    blockers: list[Blocker] = Field(default_factory=list)             # likely auto-filters (code, from the JD)

    @property
    def gaps(self) -> list[SkillMatch]:
        """Skills and experience to work on. Soft skills are never a gap: see `soft`."""
        return [m for m in self.matches if m.status != "meets" and not m.is_soft]

    @property
    def soft(self) -> dict[str, list[SkillMatch]]:
        """Ways of working the job names. 'shown' = a resume line proves it; 'interview' = nothing to
        fix on the resume, prepare a real example for the interview."""
        rows = [m for m in self.matches if m.is_soft]
        return {"shown": [m for m in rows if m.status == "meets"],
                "interview": [m for m in rows if m.status != "meets"]}

    @property
    def fit(self) -> Fit:
        must = [m for m in self.matches if m.must_have and not m.is_soft]
        return assess_fit(self.seniority, self.role_years, self.your_years,
                          sum(m.status == "meets" for m in must), len(must), self.internship_months, self.rules)

    def comparison(self) -> str:
        def label(m: SkillMatch) -> str:
            if m.is_soft:
                return "way of working: not scored, never a gap"
            base = "must-have" if m.must_have else "a plus"
            return f"{base}; related skill shown: {m.related}" if m.related else base
        return self.fit.fact() + "\n" + "\n".join(
            f"- {m.skill}: {LEVEL_NAMES[m.current]} -> {LEVEL_NAMES[m.needed]} ({label(m)}; {m.status})"
            for m in self.matches)


def prepare(llm: LLM, resume_text: str, jd_text: str, catalog: Catalog, matcher: SkillMatcher,
            guide: dict[str, SkillGuide], title: str = "", hide: list[str] | None = None,
            years: float | None = None, internship_months: float = 0) -> Prepared:
    """Steps 1-4: read the JD and resume (2 LLM calls), then compare and plan in code."""
    if not jd_text.strip():
        raise ValueError("A job description is required.")
    vocabulary = catalog.skills
    safe_jd = mask_pii(jd_text, hide)
    safe_resume = mask_pii(resume_text, hide)

    # 1. Read the JD
    draft: PostingDraft = llm.extract(PostingDraft, prompts.text("jd_read").format(
        vocabulary=", ".join(vocabulary), posting=wrap_document("posting", safe_jd)), system=prompts.READER)
    job = check_posting_draft(draft, "jd", safe_jd, "", matcher)
    dropped = [f"JD: {r}" for r in job.rejected]

    # 2. Read the resume, looking for this job's requirements by their own names
    run_matcher = matcher.extended([s.skill for s in job.skills])
    profile = read_resume(llm, resume_text, run_matcher, vocabulary, hide,
                          requirements=[f"{s.skill} ({s.kind})" for s in job.skills])
    from app.services.implied import add_implied
    profile = profile.model_copy(update={"skills": add_implied(profile.skills)})
    dropped += [f"Resume: {r.skill} ({r.reason})" for r in profile.rejected]

    # 3. Compare
    matches = compare(job.skills, profile.skills)

    # 4. Learn
    levels = {m.skill: int(m.current) for m in matches}
    for sk in profile.skills:
        levels[sk.skill] = max(levels.get(sk.skill, 0), int(sk.level))

    match = catalog.parse_title(title) if title else None
    decision = decide_seniority(match.seniority if match else None, jd_text)
    return Prepared(title=title or "Job description", safe_jd=safe_jd, safe_resume=safe_resume,
                    matches=matches, learning=plan(matches, levels, guide), levels=levels,
                    checks=resume_checks(resume_text, profile, matches),
                    other_skills=sorted({s.skill for s in profile.skills if s.source != "implied"}
                                        - {m.skill for m in matches}), dropped=dropped,
                    seniority=decision.seniority, seniority_note=decision.note,
                    role_years=years_required(jd_text), your_years=years,
                    internship_months=internship_months, rules=experience_rules(jd_text),
                    blockers=find_blockers(jd_text, resume_text, years_required(jd_text), years,
                                           experience_rules(jd_text), internship_months))


def plan(matches: list[SkillMatch], levels: dict[str, int], guide: dict[str, SkillGuide]) -> list[LearningStep]:
    """Step 4: course areas for every gap, most important first, prerequisites before."""
    gaps = [SkillGap(skill=m.skill, current=m.current, required=m.needed,
                     priority="required" if m.must_have else "important")
            for m in matches if m.status != "meets" and not m.is_soft]
    gaps.sort(key=lambda g: (PRIORITY_ORDER[g.priority], -g.size, g.skill))
    return learning_steps(gaps, {k: Level(v) for k, v in levels.items()}, guide)


# ---------- Questions about the gaps (made in code, no LLM) ----------

def gap_questions(prep: "Prepared", listed_only: set[str], limit: int = 4) -> list[GapQuestion]:
    """Ask about the most important gaps: the answer may show the student knows more than
    the resume says. Must-haves first; skills listed without proof first among those."""
    rows = sorted(prep.gaps, key=lambda m: (not m.must_have, m.skill not in listed_only, m.skill))
    rows += sorted(prep.soft["interview"], key=lambda m: m.skill)   # ways of working: asked last, only if room
    out = []
    for m in rows[:limit]:
        needed = LEVEL_NAMES[m.needed]
        if m.kind == "soft":
            out.append(GapQuestion(skill=m.skill, issue=f"The job mentions {m.skill.lower()}. No resume line proves it, and you don't need a "
                                         f"soft-skills list: one real example is enough.",
                                   question=f"Tell us about one time you showed {m.skill.lower()} (presenting, writing, "
                                            f"working with a team or stakeholders). What did you do, and what happened?"))
        elif m.related and m.status == "missing":
            out.append(GapQuestion(skill=m.skill,
                                   issue=f"The job asks for {m.skill}. Your resume shows {m.related}, a related tool: "
                                         f"that helps, but it isn't the same.",
                                   question=f"Have you used {m.skill} itself, or only {m.related}? If only {m.related}, "
                                            f"what did you build with it?"))
        elif m.skill in listed_only:
            out.append(GapQuestion(skill=m.skill,
                                   issue=f"Your resume lists {m.skill} but doesn't show where you used it.",
                                   question=f"Where have you used {m.skill}, and what exactly did you do with it?"))
        elif m.status == "below":
            out.append(GapQuestion(skill=m.skill,
                                   issue=f"The job needs {m.skill} at {needed}; your resume shows {LEVEL_NAMES[m.current]}.",
                                   question=f"Have you done more with {m.skill} than your resume shows? What, and where?"))
        elif m.kind == "task":
            out.append(GapQuestion(skill=m.skill, issue=f"The job asks for experience with {m.skill.lower()}; your "
                                                        f"resume doesn't show it.",
                                   question=f"Have you worked on {m.skill.lower()} anywhere (a course, project or job) "
                                            f"that your resume leaves out? What did you do?"))
        else:
            out.append(GapQuestion(skill=m.skill, issue=f"The job asks for {m.skill}; your resume doesn't mention it.",
                                   question=f"Have you used {m.skill} anywhere (a course, project or job) that your "
                                            f"resume leaves out? What did you do?"))
    return out


OVERCLAIM = re.compile(r"\b(strong|good|great|excellent|solid|ideal|perfect)\s+(fit|match|candidate)\b|"
                       r"\bwell[- ]suited\b|\b(?:you are|you're) (?:a )?(?:fit|ready)\b|\bqualif(?:y|ied) for\b",
                       re.IGNORECASE)


FEEDBACK_BLOCK = "\n" + prompts.text("advise_feedback") + "\n"


def advise(llm: LLM, prep: Prepared, matcher: SkillMatcher, feedback: str = ""):
    """Steps 5-6: one LLM call for advice, then code checks. Returns (draft summary, projects,
    questions, dropped). `feedback` comes from the judge on a retry (M5)."""
    fit = prep.fit
    unguided = [m.skill for m in prep.gaps if not m.in_guide]
    prompt = ADVICE_PROMPT.format(fit=fit.fact(), comparison=prep.comparison(),
                                  unguided=", ".join(unguided) or "(none)",
                                  job=wrap_document("job_description", prep.safe_jd),
                                  resume=wrap_document("resume", prep.safe_resume))
    if feedback:
        prompt += FEEDBACK_BLOCK.format(feedback=feedback)
    advice: AdviceDraft = llm.extract(AdviceDraft, prompt, system=prompts.ADVISOR)

    job_skills = {m.skill for m in prep.matches}
    projects, dropped = [], []
    for p in advice.projects:
        named = [matcher.match(s) or s for s in p.skills]
        if not quote_is_in(p.job_quote, prep.safe_jd):
            dropped.append(f"Project '{p.title}': quote not in the job description")
        elif not named or any(s not in job_skills for s in named):
            dropped.append(f"Project '{p.title}': uses skills the job doesn't ask for")
        else:
            projects.append(p.model_copy(update={"skills": named}))
    summary = advice.summary
    if fit.verdict in ("Not yet", "Stretch") and OVERCLAIM.search(summary):
        dropped.append("Summary: it called you a fit, which the facts don't support; replaced")
        summary = (f"Fit for this role: {fit.verdict}. The gaps below are what to work on while you "
                   f"apply to roles at your level.")
    from app.services.questions import line_questions
    questions = line_questions(prep.safe_resume, sorted(job_skills))
    extras = {"show_better": _checked_show_better(advice.show_better, prep, dropped),
              "learning": _checked_learning(advice.learning, unguided, dropped)}
    return summary, projects[:3], questions, dropped, extras


def _checked_show_better(items: list[ShowBetterDraft], prep: Prepared, dropped: list[str]) -> list[ShowBetter]:
    met = {m.skill.casefold(): m.skill for m in prep.matches if m.status == "meets"}
    out: list[ShowBetter] = []
    for item in items:
        skill = met.get(item.skill.strip().casefold())
        problem = rewrite_problem(item.suggestion, item.resume_line, prep.safe_jd)
        if skill is None:
            dropped.append(f"Show better '{item.skill}': not a requirement you meet")
        elif not quote_is_in(item.resume_line, prep.safe_resume):
            dropped.append(f"Show better '{item.skill}': resume line not found")
        elif problem:
            dropped.append(f"Show better '{item.skill}': {problem}")
        elif all(b.skill != skill for b in out):
            out.append(ShowBetter(skill=skill, why="context", resume_line=item.resume_line, text=item.suggestion))
    return out


def _checked_learning(items: list[LearningSuggestion], unguided: list[str],
                      dropped: list[str]) -> dict[str, LearningSuggestion]:
    wanted = {s.casefold(): s for s in unguided}
    out: dict[str, LearningSuggestion] = {}
    for item in items:
        skill = wanted.get(item.skill.strip().casefold())
        text = " ".join([item.area, *item.topics])
        if skill is None:
            dropped.append(f"Learning '{item.skill}': not a gap without a guide")
        elif PROVIDER.search(text) or not item.area.strip():
            dropped.append(f"Learning '{item.skill}': named a provider, course code or duration")
        else:
            out[skill] = item.model_copy(update={"skill": skill, "topics": item.topics[:4]})
    return out


def hidden_strengths(prep: Prepared) -> list[ShowBetter]:
    """Met requirements the resume only lists, or only implies: say where to name them."""
    out = []
    for m in prep.matches:
        if m.status != "meets":
            continue
        if m.visibility == "listed":
            out.append(ShowBetter(skill=m.skill, why="listed", resume_line=None,
                                  text=f"Your resume only lists {m.skill}. Name it in the project or job line "
                                       f"where you used it, so a recruiter sees it in use."))
        elif m.visibility == "implied":
            out.append(ShowBetter(skill=m.skill, why="implied", resume_line=None,
                                  text=f"{m.resume_quote}. Name {m.skill} in that line if you used it there; "
                                       f"screeners search for the word."))
    return out


def assemble(prep: Prepared, summary: str, projects, questions, dropped, extras: dict | None = None) -> Analysis:
    extras = extras or {}
    listed = {s for c in prep.checks if c["area"] == "Listed but not shown" for s in c.get("skills", [])}
    suggestions: dict[str, LearningSuggestion] = extras.get("learning", {})
    learning = [step.model_copy(update={"area": suggestions[step.skill].area, "topics": suggestions[step.skill].topics,
                                        "suggested": True})
                if not step.in_guide and step.skill in suggestions else step for step in prep.learning]
    context = extras.get("show_better", [])
    show = hidden_strengths(prep) + [b for b in context if b.skill not in {h.skill for h in hidden_strengths(prep)}]
    return Analysis(title=prep.title, seniority=prep.seniority, seniority_note=prep.seniority_note,
                    summary=summary, matches=prep.matches, learning=learning, show_better=show,
                    projects=projects, questions=questions, checks=prep.checks,
                    other_skills=prep.other_skills, dropped=prep.dropped + dropped,
                    gap_questions=gap_questions(prep, listed), fit=prep.fit, blockers=prep.blockers)


def analyze(llm: LLM, resume_text: str, jd_text: str, catalog: Catalog, matcher: SkillMatcher,
            guide: dict[str, SkillGuide], title: str = "", hide: list[str] | None = None) -> Analysis:
    """The whole analysis without review: 3 LLM calls."""
    prep = prepare(llm, resume_text, jd_text, catalog, matcher, guide, title, hide)
    return assemble(prep, *advise(llm, prep, matcher))
