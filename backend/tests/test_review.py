"""The judge, scored in code, with a retry on failure. All with a fake LLM."""

import pytest

from app.services.analysis import AdviceDraft, ProjectIdea
from app.services.catalog import load_catalog
from app.services.learning import load_guide
from app.services.llm import FakeLLM
from app.services.matching import load_matcher
from app.services.review import (
    Criterion, PairChoice, Verdict, analyze_with_review, pairwise, score_verdict,
)
from test_analysis import JD, RESUME, fake_llm

ADVICE = "Summary: Add Kubernetes to your skills section.\nLearning plan:\n- Take a course in RAG"


def ok(score=3):
    return Criterion(reasoning="fine", evidence="", score=score)


def verdict(**changes):
    parts = dict(gap_accuracy=ok(), no_stuffing=ok(), actionability=ok(), personalisation=ok())
    parts.update(changes)
    return Verdict(**parts)


def test_all_good_passes():
    r = score_verdict(verdict(), ADVICE)
    assert r.passed and r.total == 12


def test_stuffing_must_be_perfect():
    bad = Criterion(reasoning="Tells the student to add a skill.",
                    evidence="Add Kubernetes to your skills section", score=2)
    r = score_verdict(verdict(no_stuffing=bad), ADVICE)
    assert not r.passed and "no_stuffing" in r.failures


def test_complaint_with_invented_quote_is_ignored():
    bad = Criterion(reasoning="Generic.", evidence="Network with recruiters on LinkedIn", score=1)
    r = score_verdict(verdict(personalisation=bad), ADVICE)
    assert r.passed and r.unverified == ["personalisation"]


def good_advice():
    return AdviceDraft(summary="Needs RAG and vector databases.",
                       projects=[ProjectIdea(title="Cited Q&A", build="A RAG app.",
                                             job_quote="integrate with databases and vector databases",
                                             skills=["RAG"])])


def run(verdicts, extra_advice=()):
    catalog = load_catalog()
    llm = fake_llm()                          # JD, resume, first advice
    llm.extractions += list(extra_advice)
    judge = FakeLLM(extractions=verdicts)
    out = analyze_with_review(llm, RESUME, JD, catalog, load_matcher(catalog.skills), load_guide(),
                              "AI Engineer", judge=judge)
    return out, llm, judge


def test_pass_first_time_means_one_review():
    out, llm, judge = run([verdict()])
    assert out.review["passed"] and len(out.review["attempts"]) == 1
    assert len(llm.seen_prompts) == 3


def test_failure_retries_with_feedback():
    stuffing = Criterion(reasoning="The summary tells them to add Docker.",
                         evidence="Good Python base", score=1)
    out, llm, judge = run([verdict(no_stuffing=stuffing), verdict()], [good_advice()])
    assert [a["passed"] for a in out.review["attempts"]] == [False, True]
    assert "reviewer rejected" in llm.seen_prompts[-1] and "add Docker" in llm.seen_prompts[-1]
    assert out.summary == "Needs RAG and vector databases."


def test_keeps_best_attempt_when_both_fail():
    weak = Criterion(reasoning="Vague.", evidence="Good Python base", score=1)
    weaker = Criterion(reasoning="Vague.", evidence="Needs RAG", score=1)
    out, _, _ = run([verdict(actionability=weak),
                     verdict(actionability=weaker, personalisation=weaker)], [good_advice()])
    assert not out.review["passed"]
    assert out.summary.startswith("Good Python base")         # attempt 1 scored higher


def choice(w):
    return PairChoice(reasoning="...", winner=w)


@pytest.mark.parametrize("answers,expected", [
    (["A", "B"], "first"),        # first wins in both orders
    (["B", "A"], "second"),
    (["A", "A"], "tie"),          # always picks position A: position bias -> tie
    (["tie", "B"], "tie"),
])
def test_pairwise_swaps_order(answers, expected):
    judge = FakeLLM(extractions=[choice(a) for a in answers])
    assert pairwise(judge, "task", "answer one", "answer two") == expected
    assert judge.seen_prompts[0].index("answer one") < judge.seen_prompts[0].index("answer two")
    assert judge.seen_prompts[1].index("answer two") < judge.seen_prompts[1].index("answer one")
