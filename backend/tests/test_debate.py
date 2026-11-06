"""The debate: rankings checked in code, Borda in code, early stop, skip when few gaps."""

from app.services.analysis import prepare
from app.services.catalog import load_catalog
from app.services.debate import (
    FocusReason, Ranking, borda, clean_ranking, debate, disagreement, self_consistency,
)
from app.services.learning import load_guide
from app.services.llm import FakeLLM
from app.services.matching import load_matcher
from test_analysis import JD, RESUME, fake_llm


def prep_from(extra):
    catalog = load_catalog()
    llm = fake_llm()
    llm.extractions = llm.extractions[:2] + list(extra)       # JD + resume, then our script
    p = prepare(llm, RESUME, JD, catalog, load_matcher(catalog.skills), load_guide(), "AI Engineer")
    return p, llm


def rank(*order):
    return Ranking(reasoning="because", order=list(order))


def test_clean_ranking_drops_invented_and_fills_missing():
    gaps = ["Python", "RAG", "Docker"]
    assert clean_ranking(["rag", "Kubernetes", "RAG"], gaps) == ["RAG", "Python", "Docker"]


def test_borda_and_disagreement():
    rs = [["a", "b", "c"], ["a", "c", "b"], ["b", "a", "c"]]
    assert borda(rs) == ["a", "b", "c"]
    assert disagreement([["a", "b"], ["a", "b"]]) == 0.0
    assert disagreement([["a", "b"], ["b", "a"]]) == 1.0


def test_debate_runs_rounds_and_combines():
    prep, _ = prep_from([])
    gaps = [m.skill for m in prep.gaps]                        # Python, RAG, Docker, Vector Databases, Pandas
    assert len(gaps) >= 4
    a, b, c = gaps[0], gaps[1], gaps[2]
    script = [rank(a, b, c), rank(b, a, c), rank(c, b, a),      # round 0: disagree
              rank(b, a, c), rank(b, a, c), rank(b, c, a),      # round 1: converging
              rank(b, a, c), rank(b, a, c), rank(b, a, c),      # round 2: agree
              FocusReason(reason="Start with what unlocks the rest.")]
    llm = FakeLLM(extractions=script)
    result = debate(llm, prep, rounds=2)
    assert result.focus[0] == b and len(result.focus) == 3
    assert result.disagreement[0] > result.disagreement[-1] == 0.0
    assert result.calls == 10
    assert "Advisor 1" in llm.seen_prompts[3] and "hiring manager" not in llm.seen_prompts[3].split("Other advisors")[1]


def test_debate_stops_early_when_everyone_agrees():
    prep, _ = prep_from([])
    g = [m.skill for m in prep.gaps]
    llm = FakeLLM(extractions=[rank(*g), rank(*g), rank(*g), FocusReason(reason="r")])
    result = debate(llm, prep, rounds=2)
    assert result.calls == 4 and len(result.rounds) == 1


def test_no_debate_for_few_gaps():
    prep, _ = prep_from([])
    prep = prep.model_copy(update={"matches": prep.gaps[:2]})
    assert debate(FakeLLM(), prep) is None


def test_self_consistency_control():
    prep, _ = prep_from([])
    g = [m.skill for m in prep.gaps]
    llm = FakeLLM(extractions=[rank(g[1], g[0]), rank(g[1], g[2]), rank(g[0], g[1])])
    assert self_consistency(llm, prep, 3)[0] == g[1]
