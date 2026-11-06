"""M5: is SkillBridge actually better than just asking an LLM? Measured, not claimed.

For each labelled case (data/eval/cases.json) we run:
  SkillBridge   the full analysis (3 LLM calls)
  Baseline      one plain prompt: "here is a resume and a JD, list the gaps and advise"
both on the same masked text, `runs` times each, and measure:

  gap precision / recall / F1   against the human labels (micro-averaged over cases)
  consistency                   mean Jaccard of the gap lists across repeated runs
  stuffing rate                 share of outputs the judge fails on "no_stuffing"
  pairwise win rate             judge picks the better answer (both orders), with a
                                95% Wilson interval, because with few cases it is wide

calibrate() checks the judge itself: Cohen's kappa against human scores.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.core import prompts
from app.services.analysis import analyze
from app.services.samples import get_sample
from app.services.guardrails import mask_pii, wrap_document
from app.services.llm import LLM
from app.services.metrics import cohen_kappa, consistency, micro_prf, wilson_interval
from app.services.review import CRITERIA, Verdict, advice_text, pairwise, passes, review, score_verdict, JUDGE_PROMPT

EVAL_DIR = Path(__file__).resolve().parents[3] / "evals" / "datasets"


class BaselineDraft(BaseModel):
    gaps: list[str] = Field(description="Skills the job needs that the resume does not show")
    advice: str = Field(description="Your advice to the student")


BASELINE_PROMPT = prompts.text("baseline")


def load_cases(path: Path = EVAL_DIR / "cases.json") -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def _names(skills, matcher) -> set[str]:
    return {(matcher.match(s) or s.strip()).casefold() for s in skills if s.strip()}


def gold_facts(case: dict) -> str:
    return "Skill gaps decided by a human reviewer:\n" + "\n".join(f"- {g}" for g in case["gold_gaps"])


def run_case(llm: LLM, judge: LLM, case: dict, catalog, matcher, guide, runs: int = 3) -> dict:
    resume, jd = get_sample("resumes", case["resume"]).text, get_sample("jds", case["jd"]).text
    ignore = {s.casefold() for s in case.get("ambiguous", [])}
    sb_runs, base_runs, sb_text, base_text = [], [], "", ""
    for i in range(runs):
        a = analyze(llm, resume, jd, catalog, matcher, guide, case.get("title", ""))
        sb_runs.append({m.skill.casefold() for m in a.gaps} - ignore)
        b: BaselineDraft = llm.extract(BaselineDraft, BASELINE_PROMPT.format(
            job=wrap_document("job_description", mask_pii(jd)),
            resume=wrap_document("resume", mask_pii(resume))))
        base_runs.append(_names(b.gaps, matcher) - ignore)
        if i == 0:
            sb_text = advice_text(a.summary, a.projects, a.questions, a.learning)
            base_text = "Gaps: " + ", ".join(b.gaps) + "\n" + b.advice
    facts = gold_facts(case)
    return {
        "id": case["id"],
        "gold": {s.casefold() for s in case["gold_gaps"]} - ignore,
        "skillbridge": {"gaps": sb_runs, "review": review(judge, facts, sb_text)},
        "baseline": {"gaps": base_runs, "review": review(judge, facts, base_text)},
        "pairwise": pairwise(judge, f"Advise a student applying for: {case.get('title', 'this job')}",
                             sb_text, base_text),
    }


def summarise(results: list[dict]) -> dict:
    out = {}
    for system in ("skillbridge", "baseline"):
        p, r, f1 = micro_prf([(res[system]["gaps"][0], res["gold"]) for res in results])
        stuffed = sum(1 for res in results if "no_stuffing" in res[system]["review"].failures)
        out[system] = {"precision": p, "recall": r, "f1": f1,
                       "consistency": sum(consistency(res[system]["gaps"]) for res in results) / len(results),
                       "stuffing_rate": stuffed / len(results)}
    wins = sum(1 for res in results if res["pairwise"] == "first")
    losses = sum(1 for res in results if res["pairwise"] == "second")
    decided = wins + losses
    out["pairwise"] = {"wins": wins, "losses": losses, "ties": len(results) - decided,
                       "win_rate": wins / decided if decided else None,
                       "interval": wilson_interval(wins, decided)}
    out["cases"] = len(results)
    return out


def format_summary(s: dict) -> str:
    rows = ["metric           SkillBridge   Baseline"]
    for key in ("precision", "recall", "f1", "consistency", "stuffing_rate"):
        rows.append(f"{key:<16} {s['skillbridge'][key]:>11.2f}   {s['baseline'][key]:>8.2f}")
    pw = s["pairwise"]
    low, high = pw["interval"]
    rows.append(f"\nJudge preferred SkillBridge {pw['wins']}x, baseline {pw['losses']}x, tie {pw['ties']}x"
                f"  (95% interval for SkillBridge's win rate: {low:.2f}-{high:.2f})")
    if s["cases"] < 10:
        rows.append(f"Only {s['cases']} case(s): treat these numbers as a smoke test, not a result.")
    return "\n".join(rows)


# ---------- Is the judge trustworthy? ----------

def calibrate(judge: LLM, path: Path = EVAL_DIR / "judge_labels.json") -> dict:
    """Run the judge on human-scored advice. Report agreement per criterion (quadratic
    weighted kappa, since scores are ordered) and on overall pass/fail (plain kappa)."""
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    human, machine = {c: [] for c in CRITERIA}, {c: [] for c in CRITERIA}
    human_pass, judge_pass = [], []
    criteria = "\n".join(f"- {n}: {t}" for n, t in CRITERIA.items())
    for item in items:
        verdict: Verdict = judge.extract(Verdict, JUDGE_PROMPT.format(
            criteria=criteria, comparison=item["facts"], advice=wrap_document("advice", item["advice"])), system=prompts.JUDGE)
        for c in CRITERIA:
            human[c].append(item["human"][c])
            machine[c].append(getattr(verdict, c).score)
        human_pass.append(passes(item["human"]))
        judge_pass.append(score_verdict(verdict, item["advice"]).passed)
    return {"items": len(items),
            "kappa": {c: cohen_kappa(human[c], machine[c], "quadratic") for c in CRITERIA},
            "pass_kappa": cohen_kappa(human_pass, judge_pass),
            "pass_agreement": sum(h == j for h, j in zip(human_pass, judge_pass)) / len(items)}
