"""M6: three advisors debate what the student should work on FIRST.

Why a debate here? The gaps are facts (computed in code). What to do first is a
judgement call with real trade-offs, which is where different viewpoints help:
  hiring_manager        what gets past screening for THIS job soonest
  learning_coach        what unlocks the most later skills (prerequisites)
  portfolio_strategist  what makes the strongest project to show

How it runs (Du et al. 2023 style, with fixes from later papers):
  round 0   each advisor ranks the gaps alone (no anchoring on others)
  round 1+  each sees the OTHER rankings, anonymised ("Advisor 2"), and may revise.
            Anonymous so nobody defers to a role title. We stop early if all agree.
  combine   Borda count IN CODE (not an LLM vote), then one moderator call writes
            the reason for the top three.
Python checks every ranking: only real gaps, no duplicates, missing gaps appended.

Is it worth it? self_consistency() is the control: ONE prompt asked the same number
of times, combined the same way. If the debate can't beat it at equal compute,
the extra roles add nothing. Kendall tau distance measures how much the advisors
disagreed (no disagreement = the debate was pointless for that case).
"""

from pydantic import BaseModel, Field

from app.core import prompts
from app.services.analysis import Analysis, Prepared
from app.services.guardrails import wrap_document
from app.services.llm import LLM
from app.services.metrics import kendall_tau_distance

LENSES = {
    "hiring_manager": "You are a hiring manager for this job. Rank by what most improves the "
                      "student's chance of passing screening for THIS job soonest.",
    "learning_coach": "You are a learning coach. Rank by what unlocks the most other skills: "
                      "prerequisites first, quick wins early.",
    "portfolio_strategist": "You are a portfolio strategist. Rank by what lets the student build "
                            "and show one strong, current project quickly.",
}
MIN_GAPS = 4          # with 3 or fewer gaps, the order barely matters: no debate
TOP = 3


class Ranking(BaseModel):
    reasoning: str = Field(description="Your reasoning, written before the ranking")
    order: list[str] = Field(description="Every gap skill, most important first, names exactly as given")


class FocusReason(BaseModel):
    reason: str = Field(description="Two sentences for the student: why these first, in this order")


RANK_PROMPT = prompts.text("debate_rank")

OTHERS_BLOCK = prompts.text("debate_others")

MODERATOR_PROMPT = prompts.text("debate_moderator")


class DebateResult(BaseModel):
    focus: list[str]
    reason: str
    rounds: list[dict[str, list[str]]]    # lens -> ranking, per round
    disagreement: list[float]             # mean Kendall tau distance between advisors, per round
    calls: int


def clean_ranking(order: list[str], gaps: list[str]) -> list[str]:
    """Only real gaps, each once, in the advisor's order; anything forgotten goes at the end."""
    lookup = {g.casefold(): g for g in gaps}
    seen: list[str] = []
    for name in order:
        g = lookup.get(name.strip().casefold())
        if g and g not in seen:
            seen.append(g)
    return seen + [g for g in gaps if g not in seen]


def borda(rankings: list[list[str]]) -> list[str]:
    """Each ranking gives n-1 points to its first item, n-2 to the next ... Ties: earlier in
    the first ranking wins, so the result is deterministic."""
    first = rankings[0]
    points = {g: 0 for g in first}
    for r in rankings:
        for i, g in enumerate(r):
            points[g] += len(r) - 1 - i
    return sorted(first, key=lambda g: (-points[g], first.index(g)))


def disagreement(rankings: list[list[str]]) -> float:
    pairs = [(a, b) for i, a in enumerate(rankings) for b in rankings[i + 1:]]
    return sum(kendall_tau_distance(a, b) for a, b in pairs) / len(pairs) if pairs else 0.0


def _gap_lines(prep: Prepared) -> str:
    from app.services.learning import LEVEL_NAMES
    return "\n".join(f"- {m.skill}: {LEVEL_NAMES[m.current]} -> {LEVEL_NAMES[m.needed]}"
                     f" ({'must-have' if m.must_have else 'a plus'})" for m in prep.gaps)


def debate(llm: LLM, prep: Prepared, rounds: int = 2) -> DebateResult | None:
    gaps = [m.skill for m in prep.gaps]
    if len(gaps) < MIN_GAPS:
        return None
    job = wrap_document("job_description", prep.safe_jd)
    calls = 0
    current: dict[str, Ranking] = {}
    history: list[dict[str, list[str]]] = []
    spread: list[float] = []

    for rnd in range(rounds + 1):
        new: dict[str, Ranking] = {}
        for lens, brief in LENSES.items():
            others = ""
            if rnd > 0:
                shown = [r for name, r in current.items() if name != lens]
                others = OTHERS_BLOCK.format(rankings="\n".join(
                    f"Advisor {i}: {', '.join(r.order)}\n  because: {r.reasoning}"
                    for i, r in enumerate(shown, 1)))
            r: Ranking = llm.extract(Ranking, RANK_PROMPT.format(
                lens=brief, gaps=_gap_lines(prep), others=others, job=job), system=prompts.DEBATER)
            calls += 1
            new[lens] = r.model_copy(update={"order": clean_ranking(r.order, gaps)})
        current = new
        history.append({lens: r.order for lens, r in current.items()})
        spread.append(disagreement([r.order for r in current.values()]))
        if spread[-1] == 0:
            break                                  # everyone agrees: more rounds add nothing

    focus = borda([r.order for r in current.values()])[:TOP]
    moderator: FocusReason = llm.extract(FocusReason, MODERATOR_PROMPT.format(
        focus="\n".join(f"{i}. {g}" for i, g in enumerate(focus, 1)),
        reasons="\n".join(f"- {r.reasoning}" for r in current.values())), system=prompts.ADVISOR)
    return DebateResult(focus=focus, reason=moderator.reason, rounds=history,
                        disagreement=spread, calls=calls + 1)


def self_consistency(llm: LLM, prep: Prepared, samples: int) -> list[str]:
    """The control: one neutral prompt asked `samples` times, combined by the same Borda count.
    Use samples = number of debate calls, so both get equal compute."""
    gaps = [m.skill for m in prep.gaps]
    rankings = []
    for _ in range(samples):
        r: Ranking = llm.extract(Ranking, RANK_PROMPT.format(
            lens="You are a careful career advisor. Rank by what the student should work on first.",
            gaps=_gap_lines(prep), others="", job=wrap_document("job_description", prep.safe_jd)), system=prompts.DEBATER)
        rankings.append(clean_ranking(r.order, gaps))
    return borda(rankings)[:TOP]


def add_focus(analysis: Analysis, result: DebateResult | None) -> Analysis:
    if result:
        analysis.focus, analysis.focus_reason = result.focus, result.reason
    return analysis
