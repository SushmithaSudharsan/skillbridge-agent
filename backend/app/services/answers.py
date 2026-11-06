"""Your answers count. Each answer can:
  - RAISE a skill the resume undersells   ("I used Docker in my final-year project")
  - LOWER a skill the resume oversells    ("I'm a beginner in cloud" vs "Proficient in GCP")
  - give a better resume line              ("Owned CI/CD" -> "Wrote the CI/CD code and tests")
and every answer gets a one-line "what this means".

Flow (1 LLM call), every claim checked by Python:
  - the quote really is in THAT answer                       (no invention)
  - raising is capped at Intermediate (Advanced needs resume proof) and by the source;
    lowering follows the student's own words, because that's the most honest evidence
  - a suggested line may only use tools, names and numbers that are already in the
    original line or the answer                              (no keyword stuffing)
Then the gaps, plan and advice are rebuilt.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.core import prompts
from app.services.analysis import Analysis, Change, Prepared, plan, sort_matches
from app.services.guardrails import mask_pii, quote_is_in, rewrite_problem, wrap_document
from app.services.llm import LLM
from app.services.matching import SkillMatcher
from app.services.models import Level
from app.services.resume import MAX_LEVEL

ANSWER_CAP = Level.INTERMEDIATE      # a chat answer alone never proves Advanced

WHERE = {
    "job": "Name it in the job line where you used it, saying what it did.",
    "project": "Name it in that project's line, saying what you used it for.",
    "course": "Add it under coursework (or the degree line it belongs to).",
    "certification": "Add the certification under certifications.",
    "self_study": "Add it under a 'Learning' or projects section, not under experience.",
}


class AnswerClaim(BaseModel):
    number: int = Field(description="The number of the answer this skill comes from")
    skill: str = Field(description="Skill name, using the vocabulary when one fits")
    level: int = Field(ge=1, le=3, description=(
        "The level THE ANSWER supports, even if lower than the resume claims. "
        "1 = says beginner, only tried it, one tutorial or one course; 2 = used it in projects "
        "(plural), across coursework, in an internship, or in a job; 3 = led or used it deeply at work. "
        "'coursework and projects' is 2."))
    source: Literal["job", "project", "course", "certification", "self_study"] = Field(
        description="internship counts as job")
    evidence: str = Field(description="Exact words copied from that answer")


class AnswerNote(BaseModel):
    number: int
    takeaway: str = Field(description=(
        "One specific, encouraging sentence: what this answer means for THIS job and what to do next "
        "(e.g. name the project on the resume line). Don't label the student 'a beginner' unless they said so."))
    rewrite: str = Field(default="", description=(
        "Only for answers about a resume line: that line rewritten, more specific and honest, using "
        "ONLY facts from the line and the answer. Empty if the answer gives nothing to improve it."))


class AnswerDraft(BaseModel):
    claims: list[AnswerClaim]
    notes: list[AnswerNote] = Field(default_factory=list)


PROMPT = prompts.text("answers_read")


def _pairs(analysis: Analysis) -> list[tuple[int, str, str, str | None, str | None]]:
    """(index in analysis.asked(), question, answer, skill asked about, resume line asked about)
    for every answered question."""
    asked = [(q.question, None, q.resume_line) for q in analysis.questions] + \
            [(g.question, g.skill, None) for g in analysis.gap_questions]
    return [(i, q, analysis.answer_for(i), skill, line)
            for i, (q, skill, line) in enumerate(asked) if analysis.answer_for(i)]


def apply_answers(llm: LLM, prep: Prepared, analysis: Analysis, matcher: SkillMatcher,
                  vocabulary: list[str], guide, hide: list[str] | None = None):
    """Returns (updated Prepared, changes, notes, dropped). notes: {asked index: {takeaway,
    rewrite, effect}}. No answers -> no LLM call, nothing changes."""
    pairs = [(i, q, mask_pii(a, hide), skill, line) for i, q, a, skill, line in _pairs(analysis)]
    if not pairs:
        return prep, [], {}, []
    block = "\n".join(f"{n}. Question: {q}" + (f" (about {skill})" if skill else "")
                      + (f'\n   Resume line: "{line}"' if line else "") + f"\n   Answer: {a}"
                      for n, (_, q, a, skill, line) in enumerate(pairs, 1))
    draft: AnswerDraft = llm.extract(AnswerDraft, PROMPT.format(
        comparison=prep.comparison(), vocabulary=", ".join(vocabulary),
        answers=wrap_document("answers", block)), system=prompts.ANSWER_READER)

    current = {m.skill: m.current for m in prep.matches}
    current.update({k: Level(v) for k, v in prep.levels.items() if k not in current})
    final: dict[str, tuple[Level, AnswerClaim]] = {}
    dropped = []
    for c in draft.claims:
        if not 1 <= c.number <= len(pairs) or not quote_is_in(c.evidence, pairs[c.number - 1][2]):
            dropped.append(f"Answer: {c.skill} (quote not in your answer)")
            continue
        skill = matcher.match(c.skill)
        if skill is None:
            dropped.append(f"Answer: {c.skill} (not a skill we track yet)")
            continue
        now = current.get(skill, Level.NONE)
        said = Level(c.level)
        if said < now:
            level = said                                        # the student says less: believe them
        else:
            level = max(now, min(said, MAX_LEVEL[c.source], ANSWER_CAP))   # raising is capped
        if level != now and (skill not in final or abs(level - now) > abs(final[skill][0] - now)):
            final[skill] = (level, c)

    changes, rows = [], []
    for m in prep.matches:
        if m.skill in final:
            level, claim = final[m.skill]
            where = WHERE[claim.source] if level > m.current else (
                "Your resume claims more than you described. Reword it to match what you said "
                "(e.g. 'used' instead of 'proficient' or 'owned').")
            changes.append(Change(skill=m.skill, before=m.current, after=level, said=claim.evidence,
                                  source=claim.source, where=where, question=pairs[claim.number - 1][0]))
            m = m.model_copy(update={"current": level, "resume_quote": f"Your answer: {claim.evidence}",
                                    "visibility": "answer"})
        rows.append(m)
    levels = dict(prep.levels)
    for skill, (level, _) in final.items():
        levels[skill] = int(level)
    for m in rows:
        levels[m.skill] = int(m.current)

    notes: dict[int, dict] = {}
    for note in draft.notes:
        if not 1 <= note.number <= len(pairs):
            continue
        index, _, answer, _, line = pairs[note.number - 1]
        rewrite = note.rewrite.strip() if line else ""
        if rewrite:
            problem = rewrite_problem(rewrite, line, answer)
            if problem:
                dropped.append(f"Suggested line: {problem}")
                rewrite = ""
        notes[index] = {"takeaway": note.takeaway.strip(), "rewrite": rewrite}
    for c in changes:
        effect = f"{c.skill}: {c.before.name.title()} -> {c.after.name.title()}"
        notes.setdefault(c.question, {"takeaway": "", "rewrite": ""}).setdefault("effects", []).append(effect)

    rows = sort_matches(rows)
    updated = prep.model_copy(update={"matches": rows, "levels": levels, "learning": plan(rows, levels, guide)})
    return updated, changes, notes, dropped


def rerun_with_answers(llm: LLM, prep: Prepared, analysis: Analysis, matcher: SkillMatcher,
                       vocabulary: list[str], guide, hide: list[str] | None = None,
                       judge: LLM | None = None) -> tuple[Analysis, Prepared]:
    """Apply the answers; if any skill changed, rebuild the advice for the new gaps
    (reviewed again when a judge is given). Questions and answers are kept for the report."""
    updated, changes, notes, dropped = apply_answers(llm, prep, analysis, matcher, vocabulary, guide, hide)
    if not changes:
        analysis.notes, analysis.dropped = notes, analysis.dropped + dropped
        return analysis, prep
    from app.services.analysis import advise, assemble
    if judge is not None:
        from app.services.review import reviewed_advice
        advice, verdict = reviewed_advice(llm, judge, updated, matcher)
    else:
        advice, verdict = advise(llm, updated, matcher), None
    new = assemble(updated, *advice)
    new.questions, new.gap_questions, new.answers = analysis.questions, analysis.gap_questions, analysis.answers
    new.changes, new.notes, new.review = changes, notes, verdict
    new.dropped += dropped
    return new, updated
