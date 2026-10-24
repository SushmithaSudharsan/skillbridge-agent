"""Job-market requirements, with a fake LLM."""

from pathlib import Path

import pytest

from skillbridge.catalog import load_catalog
from skillbridge.llm import FakeLLM
from skillbridge.market import (
    MIN_POSTINGS, Posting, PostingDraft, PostingSkillDraft, aggregate, build_market,
    check_posting_draft, extract_posting, load_postings, requirements_for,
)
from skillbridge.matching import load_matcher


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture(scope="module")
def matcher(catalog):
    return load_matcher(catalog.skills)


def draft(*items):
    return PostingDraft(skills=[PostingSkillDraft(skill=s, level=l, must_have=m, evidence=e)
                                for s, l, m, e in items])


def test_sample_jds_load():
    postings = load_postings()
    peoplex = next(p for p in postings if p.id == "peoplex_ai_engineer")
    assert peoplex.title == "AI Engineer" and peoplex.label == "mid"
    assert peoplex.text.startswith("We are seeking")


def test_posting_without_title_is_an_error(tmp_path):
    (tmp_path / "x.txt").write_text("no header here", encoding="utf-8")
    with pytest.raises(ValueError, match="title"):
        load_postings(tmp_path)


def test_posting_evidence_is_checked(matcher):
    text = "Strong command of Python. Experience with vector DBs."
    result = check_posting_draft(draft(
        ("Python", 3, True, "Strong command of Python"),
        ("Vector Databases", 2, True, "vector DBs"),
        ("Kubernetes", 2, True, "Kubernetes in production"),      # invented
    ), "p1", text, "hash", matcher)
    assert [(s.skill, s.in_catalog) for s in result.skills] == [
        ("Python", True), ("Vector Databases", False)]
    assert result.rejected == ["Kubernetes: evidence not in posting"]


def test_extraction_is_cached(tmp_path, matcher, catalog):
    posting = Posting(id="p1", title="AI Engineer", text="Strong command of Python.")
    llm = FakeLLM(extractions=[draft(("Python", 3, True, "Strong command of Python"))])
    first = extract_posting(llm, posting, matcher, catalog.skills, tmp_path)
    second = extract_posting(llm, posting, matcher, catalog.skills, tmp_path)   # no LLM call left
    assert first == second
    assert len(llm.seen_prompts) == 1


def make_extraction(matcher, pid, *items):
    text = " ".join(e for *_, e in items)
    return check_posting_draft(draft(*items), pid, text, "h", matcher)


def test_aggregate_counts_shares_and_levels(matcher):
    extractions = [
        make_extraction(matcher, "a", ("Python", 2, True, "Python"), ("RAG", 1, True, "RAG")),
        make_extraction(matcher, "b", ("Python", 3, True, "Python"), ("Vector Databases", 2, True, "vector")),
        make_extraction(matcher, "c", ("Python", 2, True, "Python")),
        make_extraction(matcher, "d", ("Python", 2, True, "Python"), ("RAG", 2, True, "RAG")),
    ]
    profile = aggregate("GenAI Engineer", "mid", extractions)
    reqs = {r.skill: r for r in profile.requirements}
    assert reqs["Python"].share == 1.0 and reqs["Python"].level == 2 and reqs["Python"].priority == "required"
    assert reqs["RAG"].share == 0.5
    assert profile.not_in_catalog == {"Vector Databases": 0.25}


def test_skill_mentioned_twice_in_one_posting_counts_once(matcher):
    ex = make_extraction(matcher, "a", ("Python", 1, True, "Python"), ("python3", 3, True, "Python"))
    profile = aggregate("GenAI Engineer", "mid", [ex])
    assert profile.requirements[0].share == 1.0 and profile.requirements[0].level == 3


def test_build_market_groups_by_family_and_seniority(tmp_path, matcher, catalog):
    postings = [
        Posting(id="a", title="Senior LLM Engineer", text="Python."),
        Posting(id="b", title="Frontend Developer", text="React."),           # unsupported, skipped
    ]
    llm = FakeLLM(extractions=[draft(("Python", 3, True, "Python"))])
    market = build_market(llm, catalog, matcher, postings, tmp_path)
    assert [(m.family, m.seniority, m.n_postings) for m in market] == [("GenAI Engineer", "senior", 1)]


def test_too_few_postings_fall_back_to_curated(matcher, catalog):
    ex = make_extraction(matcher, "a", ("Python", 3, True, "Python"))
    market = [aggregate("GenAI Engineer", "mid", [ex])]
    reqs, source = requirements_for(catalog, "GenAI Engineer", "mid", market)
    assert source.startswith("curated profile (only 1 posting")
    assert reqs == catalog.profile("GenAI Engineer", "mid").requirements


def test_enough_postings_use_the_market(matcher, catalog):
    exs = [make_extraction(matcher, str(i), ("Python", 3, True, "Python")) for i in range(MIN_POSTINGS)]
    market = [aggregate("GenAI Engineer", "mid", exs)]
    reqs, source = requirements_for(catalog, "GenAI Engineer", "mid", market)
    assert source == f"job market ({MIN_POSTINGS} postings)"
    assert [r.skill for r in reqs] == ["Python"]
