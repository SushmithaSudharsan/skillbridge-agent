"""The counterfactual variants must change the surface only, never the facts."""

from app.services.guardrails import mask_pii
from app.services.samples import load_samples
from app.services.variants import CAREER_BREAK, plain_prompt, polish_prompt, shuffled_sections, with_career_break

RESUMES = load_samples("resumes")


def test_career_break_only_adds_a_gap():
    for s in RESUMES:
        v = with_career_break(s.text)
        assert v.startswith(s.text.rstrip()) and CAREER_BREAK in v


def test_shuffle_keeps_every_section_and_the_header():
    for s in RESUMES:
        body = [b for b in s.text.strip().split("\n\n") if b.strip()]
        out = [b for b in shuffled_sections(s.text).strip().split("\n\n") if b.strip()]
        assert sorted(out) == sorted(body) and out[0] == body[0]
        if len(body) > 2:
            assert out != body


def test_masking_does_not_hide_a_career_break():
    """The gap stays visible to the reader, so the runner must show it changes nothing."""
    s = RESUMES[0]
    assert "Career break" in mask_pii(with_career_break(s.text))


def test_polish_prompt_forbids_adding_facts():
    p = polish_prompt("Skills: Python")
    assert "Do not add any skill" in p and "Skills: Python" in p


def test_plain_english_prompt_keeps_facts_and_asks_for_simple_wording():
    p = plain_prompt("Skills: Python")
    assert "plain, simple English" in p and "Do not add any skill" in p and "Skills: Python" in p
