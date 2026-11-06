"""Answers count: they can raise a skill (capped, quoted), and the plan is rebuilt."""

import pytest

from app.services.analysis import AdviceDraft, advise, assemble, prepare
from app.services.answers import AnswerClaim, AnswerDraft, AnswerNote, apply_answers, rerun_with_answers, rewrite_problem
from app.services.catalog import load_catalog
from app.services.learning import load_guide
from app.services.llm import FakeLLM
from app.services.matching import load_matcher
from app.services.models import Level
from app.services.resume import ResumeDraft, ResumeSkillDraft, check_draft
from test_analysis import JD, RESUME, fake_llm

DOCKER = "I used Docker to containerise the recommender API in my final-year project"


@pytest.fixture
def parts():
    catalog = load_catalog()
    matcher, guide = load_matcher(catalog.skills), load_guide()
    llm = fake_llm()
    prep = prepare(llm, RESUME, JD, catalog, matcher, guide, "AI Engineer")
    analysis = assemble(prep, *advise(llm, prep, matcher))
    return catalog, matcher, guide, prep, analysis


def answer(analysis, skill, text):
    """Put `text` as the answer to the gap question about `skill`."""
    n = len(analysis.questions) + [g.skill for g in analysis.gap_questions].index(skill)
    analysis.answers = [""] * (len(analysis.questions) + len(analysis.gap_questions))
    analysis.answers[n] = text
    return n


# ---------- Reading the resume: degree evidence and level caps ----------

def test_levels_are_capped_by_source():
    text = "Skills: MLOps\nB.Tech in Computer Science (Specialization in AI & ML)"
    draft = ResumeDraft(skills=[
        ResumeSkillDraft(skill="MLOps", level=3, source="listed", evidence="Skills: MLOps"),
        ResumeSkillDraft(skill="Machine Learning", level=3, source="course",
                         evidence="Specialization in AI & ML")])
    levels = {s.skill: s.level for s in check_draft(draft, text, load_matcher(load_catalog().skills)).skills}
    assert levels["MLOps"] == Level.BEGINNER                  # a list can't prove more
    assert levels["Machine Learning"] == Level.INTERMEDIATE   # a degree can't prove Advanced


# ---------- Gap questions ----------

def test_gap_questions_ask_about_listed_skills_first(parts):
    *_, analysis = parts
    skills = [g.skill for g in analysis.gap_questions]
    assert skills[0] == "Pandas" and len(skills) == 4
    assert "lists Pandas but doesn't show" in analysis.gap_questions[0].issue
    assert all(g.question.endswith("?") for g in analysis.gap_questions)


# ---------- Answers raise levels, with checks ----------

def test_answer_closes_a_gap_and_rebuilds_the_plan(parts):
    catalog, matcher, guide, prep, analysis = parts
    n = answer(analysis, "Docker", DOCKER)
    llm = FakeLLM(extractions=[AnswerDraft(claims=[AnswerClaim(
        number=1, skill="docker", level=2, source="project", evidence="used Docker to containerise the recommender API")])])
    updated, changes, notes, dropped = apply_answers(llm, prep, analysis, matcher, catalog.skills, guide)
    status = {m.skill: m.status for m in updated.matches}
    assert status["Docker"] == "meets"
    assert not any(s.skill == "Docker" for s in updated.learning)
    assert any(s.skill == "Docker" for s in prep.learning)          # it was there before
    assert changes[0].after == Level.INTERMEDIATE and "project" in changes[0].where
    assert "Your answer:" in next(m for m in updated.matches if m.skill == "Docker").resume_quote
    assert n >= 0 and not dropped


def test_answers_cannot_prove_advanced(parts):
    catalog, matcher, guide, prep, analysis = parts
    answer(analysis, "Python", "I led the Python backend at work for two years")
    llm = FakeLLM(extractions=[AnswerDraft(claims=[AnswerClaim(
        number=1, skill="Python", level=3, source="job", evidence="led the Python backend at work")])])
    updated, changes, notes, _ = apply_answers(llm, prep, analysis, matcher, catalog.skills, guide)
    py = next(m for m in updated.matches if m.skill == "Python")
    assert py.current == Level.INTERMEDIATE and py.status == "below"   # Advanced needs resume proof
    assert changes == []                                                # it was already Intermediate


def test_invented_quote_is_dropped(parts):
    catalog, matcher, guide, prep, analysis = parts
    answer(analysis, "RAG", "Only a tutorial, nothing real.")
    llm = FakeLLM(extractions=[AnswerDraft(claims=[AnswerClaim(
        number=1, skill="RAG", level=2, source="project", evidence="built a production RAG system")])])
    updated, changes, notes, dropped = apply_answers(llm, prep, analysis, matcher, catalog.skills, guide)
    assert changes == [] and "quote not in your answer" in dropped[0]


def test_no_answers_means_no_llm_call(parts):
    catalog, matcher, guide, prep, analysis = parts
    analysis.answers = ["", "  "]
    llm = FakeLLM()
    assert apply_answers(llm, prep, analysis, matcher, catalog.skills, guide) == (prep, [], {}, [])
    assert llm.seen_prompts == []


def test_answers_are_masked_before_the_llm(parts):
    catalog, matcher, guide, prep, analysis = parts
    answer(analysis, "Docker", "Yes, ask my manager at priya.raman@example.com, +91 98765 43210")
    llm = FakeLLM(extractions=[AnswerDraft(claims=[])])
    apply_answers(llm, prep, analysis, matcher, catalog.skills, guide)
    assert "example.com" not in llm.seen_prompts[0] and "98765" not in llm.seen_prompts[0]


def test_rerun_rebuilds_advice_and_keeps_the_answers(parts):
    catalog, matcher, guide, prep, analysis = parts
    answer(analysis, "Docker", DOCKER)
    llm = FakeLLM(extractions=[
        AnswerDraft(claims=[AnswerClaim(number=1, skill="Docker", level=2, source="project",
                                        evidence="used Docker to containerise the recommender API")]),
        AdviceDraft(summary="Docker is covered now; RAG is the main gap.", projects=[])])
    new, new_prep = rerun_with_answers(llm, prep, analysis, matcher, catalog.skills, guide)
    assert new.summary.startswith("Docker is covered now")
    assert [c.skill for c in new.changes] == ["Docker"]
    assert new.answers == analysis.answers and new.gap_questions == analysis.gap_questions
    assert "Docker: Intermediate -> Intermediate (must-have; meets)" in llm.seen_prompts[1]
    assert new_prep.levels["Docker"] == 2


# ---------- Answers can also LOWER an oversold skill, and improve a line ----------

def test_saying_beginner_lowers_an_oversold_skill(parts):
    catalog, matcher, guide, prep, analysis = parts
    answer(analysis, "Python", "Honestly I'm only a beginner in Python, I copied most of it")
    llm = FakeLLM(extractions=[AnswerDraft(claims=[AnswerClaim(
        number=1, skill="Python", level=1, source="self_study", evidence="only a beginner in Python")])])
    updated, changes, notes, _ = apply_answers(llm, prep, analysis, matcher, catalog.skills, guide)
    py = next(m for m in updated.matches if m.skill == "Python")
    assert py.current == Level.BEGINNER                                  # was Intermediate
    assert changes[0].after < changes[0].before and "claims more" in changes[0].where
    assert any(s.skill == "Python" and s.level == Level.INTERMEDIATE for s in updated.learning)


def test_rewrite_check():
    line = "Owned CI/CD for automated retraining and PyTest testing; tracked 50+ MLflow experiments"
    said = "I wrote the code and the tests required in CI/CD."
    assert rewrite_problem("Wrote the CI/CD code and PyTest tests for automated retraining; "
                           "tracked 50+ MLflow experiments", line, said) is None
    assert "AWS" in rewrite_problem("Wrote CI/CD on AWS with PyTest", line, said)
    assert "80" in rewrite_problem("Wrote CI/CD, cutting deploy time 80%", line, said)
    assert "Kubernetes" in rewrite_problem("Wrote the CI/CD code and deployed it to Kubernetes", line, said)
    assert rewrite_problem("Wrote the tests. Tracked 50+ MLflow experiments", line, said) is None


def test_notes_attach_to_the_right_question_and_bad_rewrites_are_dropped(parts):
    catalog, matcher, guide, prep, analysis = parts
    line = analysis.questions[0].resume_line                      # the recommender line
    analysis.answers = ["It reached 0.87 R2 on a 20% test split"] + [""] * len(analysis.gap_questions)
    good = f"{line[0].upper()}{line[1:]}, reaching 0.87 R2 on a 20% test split"
    llm = FakeLLM(extractions=[AnswerDraft(claims=[], notes=[
        AnswerNote(number=1, takeaway="A measured result makes this line much stronger.", rewrite=good)])])
    _, _, notes, dropped = apply_answers(llm, prep, analysis, matcher, catalog.skills, guide)
    assert notes[0]["rewrite"] == good and notes[0]["takeaway"].startswith("A measured result")
    assert not dropped
    analysis.notes = notes
    asked = analysis.asked()
    assert asked[0]["rewrite"] == good and asked[1]["takeaway"] == ""

    llm = FakeLLM(extractions=[AnswerDraft(claims=[], notes=[
        AnswerNote(number=1, takeaway="x", rewrite="Deployed on Kubernetes, 0.87 R2")])])
    _, _, notes, dropped = apply_answers(llm, prep, analysis, matcher, catalog.skills, guide)
    assert notes[0]["rewrite"] == "" and "Kubernetes" in dropped[0]
