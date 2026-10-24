"""Command line for SkillBridge.

  python -m skillbridge.cli check                         test your LLM key (1 call)
  python -m skillbridge.cli resume my_resume.txt          read a resume (1 call)
  python -m skillbridge.cli market                        read data/sample_jds (1 call per new JD)
  python -m skillbridge.cli roadmap --student aiml_fresher --goal "Junior Data Scientist"
  python -m skillbridge.cli roadmap --resume my_resume.txt --goal "AI Engineer" --jd jd.txt
"""

import argparse
import json
from pathlib import Path

from skillbridge.catalog import DEFAULT_DATA_DIR, load_catalog
from skillbridge.evaluate import load_students
from skillbridge.llm import get_llm
from skillbridge.market import MarketProfile, build_market, load_postings
from skillbridge.matching import load_matcher
from skillbridge.resume import read_resume
from skillbridge.tools import AgentContext

MARKET_FILE = DEFAULT_DATA_DIR / "cache" / "market.json"


def cmd_check(args) -> None:
    llm = get_llm()
    turn = llm.chat([{"role": "user", "content": "Reply with the single word: ready"}], [])
    print(f"Model: {llm.model_name}\nReply: {turn.text.strip()}")


def cmd_resume(args) -> None:
    catalog = load_catalog()
    profile = read_resume(get_llm(), Path(args.file).read_text(encoding="utf-8"),
                          load_matcher(catalog.skills), catalog.skills)
    _print_profile(profile)


def cmd_market(args) -> None:
    catalog = load_catalog()
    postings = load_postings()
    print(f"Postings found: {len(postings)}")
    market = build_market(get_llm(), catalog, load_matcher(catalog.skills), postings)
    MARKET_FILE.parent.mkdir(parents=True, exist_ok=True)
    MARKET_FILE.write_text(json.dumps([m.model_dump(mode="json") for m in market], indent=2),
                           encoding="utf-8")
    for m in market:
        print(f"\n{m.family} / {m.seniority}: {m.n_postings} posting(s)")
        for r in m.requirements:
            print(f"  {r.skill:<20} {r.level.name.title():<13} {r.priority:<10} {r.share:.0%}")
        for skill, share in m.not_in_catalog.items():
            print(f"  {skill:<20} (no course yet)            {share:.0%}")
    print(f"\nSaved to {MARKET_FILE}")


def cmd_roadmap(args) -> None:
    catalog = load_catalog()
    llm = get_llm()
    if args.resume:
        profile = read_resume(llm, Path(args.resume).read_text(encoding="utf-8"),
                              load_matcher(catalog.skills), catalog.skills)
        _print_profile(profile)
        skills = profile.skills
    else:
        student = next(s for s in load_students() if s.id == args.student)
        skills = student.skills
    market = []
    if MARKET_FILE.exists():
        market = [MarketProfile.model_validate(m)
                  for m in json.loads(MARKET_FILE.read_text(encoding="utf-8"))]
    context = AgentContext(catalog=catalog, student_skills=skills, market=market)

    message = f"I want to become: {args.goal}."
    if args.jd:
        message += "\nHere is the job description:\n" + Path(args.jd).read_text(encoding="utf-8")

    if args.engine == "langchain":
        from skillbridge.lc_agent import run_langchain_agent
        roadmap, problems, _ = run_langchain_agent(llm.model_name, context, message)
    else:
        from skillbridge.agent import compose_roadmap, run_agent
        run = run_agent(llm, context, message)
        print("\n--- Tool calls ---")
        for i, step in enumerate(run.steps, 1):
            print(f"{i}. {step.tool}({json.dumps(step.args)})")
        print(f"\n--- Agent answer ({run.stopped}) ---\n{run.answer}")
        roadmap, problems = compose_roadmap(llm, run, context)

    print("\n--- Roadmap ---")
    print(roadmap.model_dump_json(indent=2))
    print("\n--- Checks ---")
    print("All checks passed" if not problems else "\n".join(f"- {p}" for p in problems))


def _print_profile(profile) -> None:
    print("--- Skills found (with proof) ---")
    for s in profile.skills:
        print(f"  {s.skill:<20} {s.level.name.title():<13} {s.confidence:<7} \"{s.evidence}\"")
    if profile.other_skills:
        print("--- Proven but not in our vocabulary ---")
        for s in profile.other_skills:
            print(f"  {s.skill}")
    if profile.rejected:
        print("--- Rejected (evidence not found in resume) ---")
        for r in profile.rejected:
            print(f"  {r.skill}: {r.reason}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="skillbridge")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check").set_defaults(func=cmd_check)
    p = sub.add_parser("resume")
    p.add_argument("file")
    p.set_defaults(func=cmd_resume)
    sub.add_parser("market").set_defaults(func=cmd_market)
    p = sub.add_parser("roadmap")
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--resume", help="path to a resume .txt file")
    who.add_argument("--student", help="id from data/test_students.json")
    p.add_argument("--goal", required=True, help='e.g. "Senior LLM Engineer"')
    p.add_argument("--jd", help="path to a job description .txt file")
    p.add_argument("--engine", choices=["hand", "langchain"], default="hand")
    p.set_defaults(func=cmd_roadmap)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
