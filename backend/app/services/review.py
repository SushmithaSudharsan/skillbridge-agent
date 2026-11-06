"""M5: an LLM judge that reviews the advice, and a retry when it fails.

Two kinds of judge:
  review()   rubric judge: scores one piece of advice on 4 criteria (1-3 each)
  pairwise() which of two answers is better, asked twice with the order swapped

Rules that keep the judge honest (same idea as the rest of SkillBridge):
  - Reasoning is written BEFORE the score, so the score follows from the reasons.
  - Pass/fail is decided in Python from the scores, not by the judge.
  - To fail a criterion the judge must quote the advice. A complaint that quotes
    text which isn't in the advice is ignored and recorded as "unverified".
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.core import prompts
from app.services.analysis import Analysis, advise, assemble, prepare
from app.services.guardrails import quote_is_in, wrap_document
from app.services.llm import LLM

CRITERIA = {
    "gap_accuracy": "Do the stated fit, gaps and strengths match the facts (including the computed overall "
                    "fit)? Nothing missing, nothing invented.",
    "no_stuffing": "Does the advice avoid telling the student to add skills, tools, numbers or employers "
                   "the resume does not show? Asking a question is fine; telling them to add is not.",
    "actionability": "Could the student act on every point today: clear course areas, concrete projects, "
                     "specific questions?",
    "personalisation": "Is the advice about THIS resume and THIS job, rather than generic career tips?",
}
PASS_SCORE = 2          # every criterion needs at least 2 of 3 ...
STRICT = {"no_stuffing": 3}   # ... and stuffing must be fully clean


def passes(scores: dict[str, int]) -> bool:
    """The pass rule, in one place (used for the judge AND for human labels)."""
    return all(scores[name] >= STRICT.get(name, PASS_SCORE) for name in CRITERIA)


class Criterion(BaseModel):
    reasoning: str = Field(description="Two or three sentences of reasoning, written before the score")
    evidence: str = Field(description="Exact words copied from the ADVICE that support the score; "
                                      "empty if the criterion is fully met")
    score: int = Field(ge=1, le=3, description="1 = fails, 2 = acceptable, 3 = fully meets")


class Verdict(BaseModel):
    gap_accuracy: Criterion
    no_stuffing: Criterion
    actionability: Criterion
    personalisation: Criterion


JUDGE_PROMPT = prompts.text("judge_rubric")


def advice_text(summary: str, projects, questions, learning) -> str:
    """The advice as plain text: what the judge reads and what quotes are checked against."""
    lines = [f"Summary: {summary}", "Learning plan:"]
    lines += [f"- {s.text}" for s in learning]
    lines.append("Portfolio projects:")
    lines += [f"- {p.title}: {p.build}" for p in projects]
    lines.append("Resume questions:")
    lines += [f"- {q.question}" for q in questions]
    return "\n".join(lines)


class Review(BaseModel):
    passed: bool
    scores: dict[str, int]
    failures: dict[str, str]          # criterion -> reasoning, only verified failures
    unverified: list[str]             # failing criteria whose quote wasn't in the advice

    @property
    def total(self) -> int:
        return sum(self.scores.values())

    def feedback(self) -> str:
        return "\n".join(f"- {name}: {why}" for name, why in self.failures.items())


def score_verdict(verdict: Verdict, advice: str) -> Review:
    """Turn the judge's verdict into pass/fail, in code."""
    scores, failures, unverified = {}, {}, []
    for name in CRITERIA:
        c: Criterion = getattr(verdict, name)
        needed = STRICT.get(name, PASS_SCORE)
        if c.score < needed and not (c.evidence.strip() and quote_is_in(c.evidence, advice)):
            unverified.append(name)           # a complaint we can't check doesn't count
            scores[name] = needed
            continue
        scores[name] = c.score
        if c.score < needed:
            failures[name] = c.reasoning
    return Review(passed=passes(scores), scores=scores, failures=failures, unverified=unverified)


def review(judge: LLM, facts: str, advice: str) -> Review:
    """facts: the comparison the advice must agree with (prep.comparison(), or human labels)."""
    criteria = "\n".join(f"- {name}: {text}" for name, text in CRITERIA.items())
    verdict: Verdict = judge.extract(Verdict, JUDGE_PROMPT.format(
        criteria=criteria, comparison=facts, advice=wrap_document("advice", advice)), system=prompts.JUDGE)
    return score_verdict(verdict, advice)


def reviewed_advice(llm: LLM, judge: LLM, prep, matcher, max_attempts: int = 2):
    """advise -> review; if the review fails, advise again with the judge's feedback.
    Keeps the best-scoring attempt. Returns (advice, review dict)."""
    best, best_review, history, feedback = None, None, [], ""
    for attempt in range(1, max_attempts + 1):
        advice = advise(llm, prep, matcher, feedback)
        result = review(judge, prep.comparison(), advice_text(advice[0], advice[1], advice[2], prep.learning))
        history.append({"attempt": attempt, "passed": result.passed, "scores": result.scores})
        if best_review is None or result.total > best_review.total:
            best, best_review = advice, result
        if result.passed:
            break
        feedback = result.feedback()
    return best, {"passed": best_review.passed, "scores": best_review.scores,
                  "failures": best_review.failures, "unverified": best_review.unverified,
                  "attempts": history}


def analyze_with_review(llm: LLM, resume_text: str, jd_text: str, catalog, matcher, guide,
                        title: str = "", hide: list[str] | None = None, judge: LLM | None = None,
                        max_attempts: int = 2) -> Analysis:
    """The whole analysis with review. LLM calls: 2 + up to 2 x max_attempts."""
    prep = prepare(llm, resume_text, jd_text, catalog, matcher, guide, title, hide)
    advice, verdict = reviewed_advice(llm, judge or llm, prep, matcher, max_attempts)
    analysis = assemble(prep, *advice)
    analysis.review = verdict
    return analysis


# ---------- Pairwise: which of two answers is better? ----------

class PairChoice(BaseModel):
    reasoning: str = Field(description="Compare both answers against the criteria first")
    winner: Literal["A", "B", "tie"]


PAIR_PROMPT = prompts.text("judge_pairwise")


def pairwise(judge: LLM, task: str, first: str, second: str) -> Literal["first", "second", "tie"]:
    """Ask twice, swapping positions. Only a win in BOTH orders counts; otherwise it's a tie.
    This cancels position bias (judges tend to prefer whichever answer comes first)."""
    criteria = "\n".join(f"- {name}: {text}" for name, text in CRITERIA.items())

    def ask(a: str, b: str) -> str:
        return judge.extract(PairChoice, PAIR_PROMPT.format(
            criteria=criteria, task=task, a=wrap_document("answer_A", a),
            b=wrap_document("answer_B", b)), system=prompts.JUDGE).winner

    one, two = ask(first, second), ask(second, first)
    if one == "A" and two == "B":
        return "first"
    if one == "B" and two == "A":
        return "second"
    return "tie"
