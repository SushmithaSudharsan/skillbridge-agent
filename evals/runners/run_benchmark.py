"""Evaluation gate: SkillBridge vs a plain-prompt baseline on evals/datasets/cases.json.

    python evals/runners/run_benchmark.py --runs 2 --min-precision 0.7 --min-recall 0.6 --max-cost 0.02

Exits with code 1 if SkillBridge falls below a threshold, so CI can block a prompt or model
change that makes things worse. Costs real API calls (about 4 per case per run).
"""

import argparse
import json
import sys

from app.core.logging import setup_logging
from app.services.benchmark import format_summary, load_cases, run_case, summarise
from app.services.catalog import load_catalog
from app.services.learning import load_guide
from app.services.llm import get_llm
from app.services.matching import load_matcher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="only the first N cases (0 = all)")
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--max-cost", type=float, default=1.0, help="max USD per SkillBridge analysis")
    args = parser.parse_args()
    setup_logging("WARNING")

    catalog = load_catalog()
    llm, judge = get_llm(), get_llm(judge=True)
    cases = load_cases()[: args.limit or None]
    results = [run_case(llm, judge, case, catalog, load_matcher(catalog.skills), load_guide(), args.runs)
               for case in cases]
    summary = summarise(results)
    analyses = len(cases) * args.runs
    cost = getattr(llm, "total_cost", 0.0) / max(analyses, 1) / 2      # llm also ran the baseline
    print(format_summary(summary))
    print(f"\nApprox. cost per analysis: ${cost:.4f}")
    sb = summary["skillbridge"]
    failures = []
    if sb["precision"] < args.min_precision:
        failures.append(f"precision {sb['precision']:.2f} < {args.min_precision}")
    if sb["recall"] < args.min_recall:
        failures.append(f"recall {sb['recall']:.2f} < {args.min_recall}")
    if cost > args.max_cost:
        failures.append(f"cost ${cost:.4f} > ${args.max_cost}")
    print(json.dumps({"skillbridge": sb, "baseline": summary["baseline"], "cost_per_analysis": cost,
                      "cases": summary["cases"], "gate_failures": failures}, default=str))
    if failures:
        print("EVAL GATE FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
