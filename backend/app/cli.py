"""Command line for SkillBridge (the web app uses the same services).

  python -m app.cli analyze --resume my_resume.pdf --jd job.txt --years 0 --report report.pdf
  python -m app.cli analyze ... --review --debate   add the LLM judge and the advisor debate
  python -m app.cli chat --resume my_resume.pdf --jd job.txt   follow-up questions
  python -m app.cli benchmark --runs 3     SkillBridge vs a plain LLM prompt
  python -m app.cli calibrate              does the judge agree with human labels?
  python -m app.cli check                  test your LLM key (1 call)
  python -m app.cli privacy my_resume.pdf  show exactly what would be sent (no LLM call)
"""

import argparse
from pathlib import Path

from app.services.catalog import load_catalog
from app.services.documents import DocumentError, read_document
from app.services.guardrails import mask_pii
from app.services.learning import LEVEL_NAMES
from app.services.llm import get_llm
from app.services.matching import load_matcher


def cmd_analyze(args) -> None:
    from app.services.analysis import advise, assemble, prepare
    from app.services.answers import rerun_with_answers
    from app.services.learning import load_guide

    catalog, guide = load_catalog(), load_guide()
    llm, matcher, hide = get_llm(), load_matcher(catalog.skills), _hidden(args)
    judge = get_llm(judge=True) if args.review else None

    years = args.years if args.years is not None else _ask_number(
        "Years of full-time work experience? (not internships or projects; Enter = 0): ")
    months = args.internship_months if args.internship_months is not None else _ask_number(
        "Months of internship experience? (Enter = 0): ")
    prep = prepare(llm, read_document(args.resume), read_document(args.jd), catalog, matcher,
                   guide, args.title or "", hide, years, months)
    if judge:
        from app.services.review import reviewed_advice
        advice, verdict = reviewed_advice(llm, judge, prep, matcher)
    else:
        advice, verdict = advise(llm, prep, matcher), None
    result = assemble(prep, *advice)
    result.review = verdict

    _print_reading(prep, result)
    _print_result(result)

    if not args.no_questions and _ask_questions(result):
        print("\nReading your answers ...")
        result, prep = rerun_with_answers(llm, prep, result, matcher, catalog.skills, guide, hide, judge)
        print("\n--- What your answers mean ---")
        for q in result.asked():
            if not q["answer"]:
                continue
            print(f"\n  Q: {q['question']}")
            if q["takeaway"]:
                print(f"     {q['takeaway']}")
            for e in q["effects"]:
                print(f"     Changed: {e}")
            if q["rewrite"]:
                print(f'     Suggested line: "{q["rewrite"]}"')
        if result.changes:
            print("\n--- Updated plan ---")
            _print_result(result, short=True)
        else:
            print("\nNo skill levels changed, so your plan stays the same.")

    if args.debate:
        from app.services.debate import add_focus, debate
        add_focus(result, debate(llm, prep))
        if result.focus:
            print("\n--- Start here ---\n  " + ", ".join(result.focus) + f"\n  {result.focus_reason}")

    path = args.report or _ask_report_path()
    if path:
        saved = save_report(result, path)
        print(f"\nReport saved to {saved} (only where you asked; SkillBridge keeps no copy)")


def _print_reading(prep, result) -> None:
    """What we understood from the resume, so the student can spot a misreading."""
    print("\n--- How we read your resume (for this job) ---")
    for m in sorted(prep.matches, key=lambda m: m.skill):
        seen = f'"{m.resume_quote}"' if m.resume_quote else "not found"
        print(f"  {m.skill:<20} {LEVEL_NAMES[m.current]:<13} {seen}")
    others = sorted(k for k, v in prep.levels.items() if v and k not in {m.skill for m in prep.matches})
    if others:
        print("  Also found: " + ", ".join(f"{k} ({LEVEL_NAMES[prep.levels[k]]})" for k in others))
    if result.dropped:
        print(f"  ({len(result.dropped)} AI claim(s) removed by the checks:)")
        for d in result.dropped:
            print(f"    - {d}")


def _ask_number(prompt: str) -> float:
    reply = _ask(prompt)
    try:
        return float(reply) if reply else 0.0
    except ValueError:
        print("  Not a number; treating it as 0.")
        return 0.0


def _print_result(result, short: bool = False) -> None:
    if not short:
        print(f"\n{result.title}" + (f" · {result.seniority}" if result.seniority else ""))
        if result.fit:
            print(f"\nFIT: {result.fit.verdict.upper()}. {result.fit.advice}")
            for r in result.fit.reasons:
                print(f"  - {r}")
        print(f"\n{result.summary}\n")
    print("--- Learn: what's missing ---")
    for m in result.gaps:
        print(f"  {m.skill:<20} {m.status:<8} {LEVEL_NAMES[m.current]} -> {LEVEL_NAMES[m.needed]}"
              f"{'  (must-have)' if m.must_have else ''}")
    if result.show_better:
        print("\n--- Show better: you have it, your resume hides it ---")
        for b in result.show_better:
            print(f"  {b.skill}: {b.text}")
    if result.strengths:
        print("\n--- Strengths ---\n  " + ", ".join(m.skill for m in result.strengths))
    print("\n--- Learning plan ---")
    for i, s in enumerate(result.learning, 1):
        print(f"  {i}. {s.text}" + (f": {', '.join(s.topics)}" if s.topics else ""))
    if short:
        return
    print("\n--- Portfolio projects ---")
    for p in result.projects:
        print(f"  - {p.title}: {p.build}")
    print("\n--- Resume ---")
    for c in result.checks:
        print(f"  {'Pass  ' if c['ok'] else 'Review'} {c['area']}: {c['text']}")
    if result.review:
        r = result.review
        print(f"\nReviewer: {'passed' if r['passed'] else 'flagged'} after {len(r['attempts'])} draft(s); "
              f"scores {r['scores']}")


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt, OSError):   # no keyboard (e.g. under pytest): skip
        return ""


def _ask_questions(result) -> bool:
    """Ask the resume questions and the gap questions. Answers are masked like the resume,
    then used to update your skill levels and plan. Returns True if anything was answered."""
    asked = [(f'Your line: "{q.resume_line}"', q.issue, q.question) for q in result.questions]
    asked += [(f"About {g.skill}", g.issue, g.question) for g in result.gap_questions]
    if not asked:
        return False
    print("\n--- A few questions: your answers can change your plan ---")
    print("Answer honestly in your own words; only what you write here can raise a skill level.")
    print("Answers are masked like your resume and never stored. Press Enter to skip.")
    answers = []
    for i, (topic, issue, question) in enumerate(asked, 1):
        print(f"\n{i}. {topic}\n   {issue}\n   {question}")
        answers.append(_ask("   your answer> "))
    result.answers = answers
    return any(answers)


def _ask_report_path() -> str:
    reply = _ask("\nSave your report as a PDF? Press Enter for report.pdf, type a file name, or n for no: ")
    if reply.lower() in {"n", "no"}:
        return ""
    return reply or "report.pdf"


def save_report(result, path: str) -> Path:
    """Write the report where the user asked. PDF unless the name ends in .html or .md."""
    from app.services.report import render_html, render_markdown

    out = Path(path)
    if out.suffix.lower() not in {".pdf", ".html", ".md"}:
        out = out.with_suffix(out.suffix + ".pdf") if out.suffix else out.with_suffix(".pdf")
    if out.suffix.lower() == ".pdf":
        from app.services.report_pdf import render_pdf
        out.write_bytes(render_pdf(result))
    elif out.suffix.lower() == ".html":
        out.write_text(render_html(result), encoding="utf-8")
    else:
        out.write_text(render_markdown(result), encoding="utf-8")
    return out


def cmd_chat(args) -> None:
    from app.services.analysis import advise, assemble, prepare
    from app.services.chat import ChatSession
    from app.services.learning import load_guide

    catalog = load_catalog()
    llm, matcher = get_llm(), load_matcher(catalog.skills)
    prep = prepare(llm, read_document(args.resume), read_document(args.jd), catalog, matcher,
                   load_guide(), args.title or "", _hidden(args), args.years, args.internship_months or 0)
    analysis = assemble(prep, *advise(llm, prep, matcher))
    print(f"{analysis.summary}\n\nAsk about your gaps, what to do first, or how to word a skill "
          f"you've used. Type 'quit' to end (the chat is then forgotten).\n")
    with ChatSession(llm, prep, analysis) as chat:
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in {"quit", "exit", "q"}:
                break
            if text:
                print(f"skillbridge> {chat.ask(text)}\n")
    print("Chat ended. Nothing was saved.")


def cmd_benchmark(args) -> None:
    from app.services.benchmark import format_summary, load_cases, run_case, summarise
    from app.services.learning import load_guide

    catalog = load_catalog()
    llm, judge = get_llm(), get_llm(judge=True)
    matcher, guide = load_matcher(catalog.skills), load_guide()
    results = []
    for case in load_cases():
        print(f"Running {case['id']} ({args.runs} runs each) ...")
        results.append(run_case(llm, judge, case, catalog, matcher, guide, args.runs))
    print("\n" + format_summary(summarise(results)))


def cmd_calibrate(args) -> None:
    from app.services.benchmark import calibrate

    out = calibrate(get_llm(judge=True))
    print(f"{out['items']} human-labelled items")
    for name, k in out["kappa"].items():
        print(f"  {name:<16} weighted kappa {k:.2f}")
    print(f"  pass/fail        kappa {out['pass_kappa']:.2f}  (raw agreement {out['pass_agreement']:.0%})")
    if out["items"] < 20:
        print("Fewer than 20 labels: too few to trust. Label more in evals/datasets/judge_labels.json.")


def cmd_check(args) -> None:
    llm = get_llm()
    turn = llm.chat([{"role": "user", "content": "Reply with the single word: ready"}], [])
    print(f"Model: {llm.model_name}\nReply: {turn.text.strip()}")


def _hidden(args) -> list[str]:
    return [t for t in (args.hide or "").split(",") if t.strip()]


def cmd_privacy(args) -> None:
    print(mask_pii(read_document(args.file), _hidden(args)))


def main() -> None:
    parser = argparse.ArgumentParser(prog="skillbridge")
    sub = parser.add_subparsers(dest="command", required=True)
    hide_help = 'extra words to hide, comma separated, e.g. "Whirl Data,Boston"'
    p = sub.add_parser("analyze", help="compare a resume with a job description")
    p.add_argument("--resume", required=True, help="path to a resume (.pdf, .docx or .txt)")
    p.add_argument("--jd", required=True, help="path to a job description (.pdf, .docx or .txt)")
    p.add_argument("--title", help='the job title, e.g. "Senior LLM Engineer"')
    p.add_argument("--hide", help=hide_help)
    p.add_argument("--report", help="save the report here: report.pdf (default type), .html or .md")
    p.add_argument("--no-questions", action="store_true", help="don't ask the resume questions")
    p.add_argument("--years", type=float, help="your years of full-time work (0 for a fresher)")
    p.add_argument("--internship-months", type=float, help="your months of internship experience")
    p.add_argument("--review", action="store_true", help="an LLM judge checks the advice; retries once")
    p.add_argument("--debate", action="store_true", help="three advisors agree on what to do first")
    p.set_defaults(func=cmd_analyze)
    p = sub.add_parser("chat", help="analyse, then ask follow-up questions")
    p.add_argument("--years", type=float, help="your years of full-time work (0 for a fresher)")
    p.add_argument("--internship-months", type=float, help="your months of internship experience")
    p.add_argument("--resume", required=True)
    p.add_argument("--jd", required=True)
    p.add_argument("--title")
    p.add_argument("--hide", help=hide_help)
    p.set_defaults(func=cmd_chat)
    p = sub.add_parser("benchmark", help="SkillBridge vs a plain LLM on evals/datasets/cases.json")
    p.add_argument("--runs", type=int, default=3)
    p.set_defaults(func=cmd_benchmark)
    sub.add_parser("calibrate", help="judge vs human labels").set_defaults(func=cmd_calibrate)
    sub.add_parser("check", help="test your LLM key (1 call)").set_defaults(func=cmd_check)
    p = sub.add_parser("privacy", help="show exactly what would be sent to the LLM (no LLM call)")
    p.add_argument("file")
    p.add_argument("--hide", help=hide_help)
    p.set_defaults(func=cmd_privacy)
    args = parser.parse_args()
    try:
        args.func(args)
    except DocumentError as error:
        raise SystemExit(f"Error: {error}") from None


if __name__ == "__main__":
    main()
