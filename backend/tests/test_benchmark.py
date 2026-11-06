"""Head-to-head benchmark and judge calibration, with fake LLMs."""

from app.services.benchmark import BaselineDraft, calibrate, format_summary, load_cases, run_case, summarise
from app.services.catalog import load_catalog
from app.services.learning import load_guide
from app.services.llm import FakeLLM
from app.services.matching import load_matcher
from app.services.review import Criterion, PairChoice, Verdict
from test_analysis import fake_llm


def ok(score=3, evidence=""):
    return Criterion(reasoning="r", evidence=evidence, score=score)


def verdict(**changes):
    parts = dict(gap_accuracy=ok(), no_stuffing=ok(), actionability=ok(), personalisation=ok())
    parts.update(changes)
    return Verdict(**parts)


BASELINE = [BaselineDraft(gaps=["RAG", "Docker", "Kubernetes"], advice="Add Kubernetes and LangChain to your resume."),
            BaselineDraft(gaps=["RAG", "vector DBs"], advice="Learn more.")]


def test_head_to_head():
    catalog = load_catalog()
    llm = FakeLLM(extractions=fake_llm().extractions + [BASELINE[0]] + fake_llm().extractions + [BASELINE[1]])
    judge = FakeLLM(extractions=[
        verdict(),                                                        # SkillBridge passes
        verdict(no_stuffing=ok(1, "Add Kubernetes and LangChain to your resume")),  # baseline stuffs
        PairChoice(reasoning="r", winner="A"), PairChoice(reasoning="r", winner="B"),  # SB wins both orders
    ])
    res = run_case(llm, judge, load_cases()[0], catalog, load_matcher(catalog.skills), load_guide(), runs=2)
    s = summarise([res])

    assert s["skillbridge"]["precision"] == 1.0                  # every SB gap is a real gap
    assert s["baseline"]["precision"] < s["skillbridge"]["precision"]   # Kubernetes is invented
    assert s["skillbridge"]["consistency"] == 1.0               # same gaps both runs
    assert s["baseline"]["consistency"] < 1.0
    assert s["baseline"]["stuffing_rate"] == 1.0 and s["skillbridge"]["stuffing_rate"] == 0.0
    assert s["pairwise"]["wins"] == 1
    assert "smoke test" in format_summary(s)


def test_baseline_sees_only_masked_text():
    catalog = load_catalog()
    llm = FakeLLM(extractions=fake_llm().extractions + [BASELINE[1]])
    judge = FakeLLM(extractions=[verdict(), verdict(), PairChoice(reasoning="r", winner="tie"),
                                 PairChoice(reasoning="r", winner="tie")])
    run_case(llm, judge, load_cases()[0], catalog, load_matcher(catalog.skills), load_guide(), runs=1)
    assert "priya.raman@example.com" not in llm.seen_prompts[-1]


def test_calibrate_perfect_judge():
    # A judge that gives exactly the human scores agrees perfectly
    import json
    from app.services.benchmark import EVAL_DIR
    items = json.loads((EVAL_DIR / "judge_labels.json").read_text())["items"]
    judge = FakeLLM(extractions=[Verdict(**{c: ok(s, "Summary") for c, s in it["human"].items()}) for it in items])
    out = calibrate(judge)
    assert out["items"] == 4 and out["pass_agreement"] == 1.0
    assert all(k == 1.0 for k in out["kappa"].values())
