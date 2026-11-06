"""Style fairness with a real LLM: the same facts written differently should get the same gaps.

    python evals/runners/run_fairness.py --runs 2 --min-jaccard 0.7

Pairs come from cases.json entries with "twin_of" (e.g. a detailed resume and a concise one
written from the same facts). Identity fairness doesn't need an LLM: masking makes the input
identical, which backend/tests/test_fairness.py proves for every sample.
"""

import argparse
import sys

from app.core.logging import setup_logging
from app.services.analysis import analyze
from app.services.benchmark import load_cases
from app.services.catalog import load_catalog
from app.services.learning import load_guide
from app.services.llm import get_llm
from app.services.matching import load_matcher
from app.services.metrics import jaccard
from app.services.samples import get_sample
from app.services.variants import Polished, plain_prompt, polish_prompt, shuffled_sections, with_career_break


def run_one(llm, case, catalog, resume: str | None = None):
    resume = resume if resume is not None else get_sample("resumes", case["resume"]).text
    jd = get_sample("jds", case["jd"]).text
    return analyze(llm, resume, jd, catalog, load_matcher(catalog.skills), load_guide(), case.get("title", ""))


def gaps(llm, case, catalog) -> set[str]:
    return {m.skill.casefold() for m in run_one(llm, case, catalog).gaps}


def variant_check(llm, cases, catalog, only: str | None) -> tuple[list[float], list[bool]]:
    """Same facts, different surface: career break, shuffled sections, AI-polished wording.
    Compares gaps (Jaccard) and the fit verdict against the original resume."""
    overlaps, same = [], []
    for case in cases.values():
        if "twin_of" in case or (only and case["id"] != only):
            continue
        text = get_sample("resumes", case["resume"]).text
        base = run_one(llm, case, catalog)
        variants = {"career-break": with_career_break(text), "shuffled": shuffled_sections(text),
                    "ai-polished": llm.extract(Polished, polish_prompt(text)).text,
                    "plain-english": llm.extract(Polished, plain_prompt(text)).text}
        for name, v in variants.items():
            other = run_one(llm, case, catalog, v)
            j = jaccard({m.skill.casefold() for m in base.gaps}, {m.skill.casefold() for m in other.gaps})
            overlaps.append(j)
            same.append(base.fit.verdict == other.fit.verdict)
            print(f"{case['id']} / {name}: gap overlap {j:.2f}, verdict {base.fit.verdict} -> {other.fit.verdict}")
    return overlaps, same


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--min-jaccard", type=float, default=0.0)
    parser.add_argument("--variants", action="store_true",
                        help="also test career break, shuffled sections, AI-polished and plain-English wording")
    parser.add_argument("--min-same-verdict", type=float, default=0.0)
    parser.add_argument("--only", default=None, help="run the variants for one case id (saves cost)")
    args = parser.parse_args()
    setup_logging("WARNING")
    catalog, llm = load_catalog(), get_llm()
    cases = {c["id"]: c for c in load_cases()}
    scores = []
    for case in cases.values():
        if "twin_of" not in case:
            continue
        twin = cases[case["twin_of"]]
        for _ in range(args.runs):
            score = jaccard(gaps(llm, case, catalog), gaps(llm, twin, catalog))
            scores.append(score)
            print(f"{case['id']} vs {twin['id']}: gap overlap (Jaccard) {score:.2f}")
    mean = sum(scores) / len(scores) if scores else 1.0
    print(f"Mean style agreement: {mean:.2f} over {len(scores)} comparison(s)")
    failed = mean < args.min_jaccard
    if failed:
        print(f"FAIRNESS GATE FAILED: {mean:.2f} < {args.min_jaccard}", file=sys.stderr)
    if args.variants:
        overlaps, same = variant_check(llm, cases, catalog, args.only)
        v_mean = sum(overlaps) / len(overlaps) if overlaps else 1.0
        v_same = sum(same) / len(same) if same else 1.0
        print(f"Variants: mean gap overlap {v_mean:.2f}, same verdict {v_same:.0%} over {len(same)} comparison(s)")
        if v_mean < args.min_jaccard or v_same < args.min_same_verdict:
            print("FAIRNESS GATE FAILED on variants", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
