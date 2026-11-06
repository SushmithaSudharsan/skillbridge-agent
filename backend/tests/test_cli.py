"""The CLI with --review and --debate, end to end, with fake LLMs (no key, no cost)."""

import builtins
import sys

from app import cli
from app.services.catalog import DEFAULT_DATA_DIR
from app.services.debate import FocusReason, Ranking
from app.services.llm import FakeLLM
from app.services.review import Criterion, Verdict
from test_analysis import fake_llm


def test_analyze_with_review_and_debate(tmp_path, capsys, monkeypatch):
    ok = Criterion(reasoning="r", evidence="", score=3)
    judge_llm = FakeLLM(extractions=[Verdict(gap_accuracy=ok, no_stuffing=ok, actionability=ok, personalisation=ok)])
    main = fake_llm()
    gaps = ["Python", "RAG", "Docker", "Vector Databases", "Pandas"]
    from app.services.answers import AnswerDraft
    main.extractions += [AnswerDraft(claims=[])]          # reading the typed answers: nothing new
    main.extractions += [Ranking(reasoning="r", order=gaps)] * 3 + [FocusReason(reason="Python unlocks the rest.")]
    monkeypatch.setattr(cli, "get_llm", lambda judge=False: judge_llm if judge else main)

    typed = iter(["About 0.87 R2 on a 20% split."])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(typed, ""))
    report = tmp_path / "report.pdf"
    monkeypatch.setattr(sys, "argv", ["skillbridge", "analyze",
        "--resume", str(DEFAULT_DATA_DIR / "sample_resumes" / "aiml_fresher.txt"),
        "--jd", str(DEFAULT_DATA_DIR / "sample_jds" / "peoplex_ai_engineer.txt"),
        "--title", "AI Engineer", "--years", "0", "--internship-months", "0", "--review", "--debate", "--report", str(report)])
    cli.main()
    out = capsys.readouterr().out
    assert "Start here" in out and "Reviewer: passed" in out
    assert "your answers can change your plan" in out and "How we read your resume" in out
    from test_report_pdf import pdf_text
    text = pdf_text(report.read_bytes())
    assert "Start here" in text and "Python unlocks the rest." in text
    assert "About 0.87 R2 on a 20% split." in text
    assert "Checked by an automatic reviewer" in text


def test_report_name_without_extension_becomes_pdf(tmp_path):
    from test_analysis import JD, RESUME
    from app.services.analysis import analyze
    from app.services.catalog import load_catalog
    from app.services.learning import load_guide
    from app.services.matching import load_matcher
    catalog = load_catalog()
    a = analyze(fake_llm(), RESUME, JD, catalog, load_matcher(catalog.skills), load_guide(), "AI Engineer")
    saved = cli.save_report(a, str(tmp_path / "my_report"))
    assert saved.name == "my_report.pdf" and saved.read_bytes().startswith(b"%PDF")
