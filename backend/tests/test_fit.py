"""Fit is decided in code from level and skills; the LLM can't overrule it.
Questions are the same every run. Every LLM call gets a persona as a system message."""

import pytest

from app.core import prompts
from app.services.analysis import AdviceDraft, advise, assemble, prepare
from app.services.catalog import load_catalog
from app.services.fit import assess_fit
from app.services.learning import load_guide
from app.services.matching import load_matcher
from app.services.questions import line_questions
from test_analysis import JD, RESUME, fake_llm


@pytest.mark.parametrize("role, years, you, met, total, verdict", [
    ("senior", 5, 0, 9, 10, "Not yet"),        # a fresher vs a senior role, however many skills
    ("mid", 3, 0, 9, 10, "Stretch"),
    ("fresher", None, 0, 9, 10, "Strong fit"),
    ("fresher", None, 0, 6, 10, "Possible fit"),
    ("fresher", None, 0, 2, 10, "Stretch"),
    ("mid", 2, 3, 8, 10, "Strong fit"),
    (None, None, 0, 9, 10, "Strong fit"),      # JD gives no level: skills decide
], ids=["fresher-vs-senior", "fresher-vs-mid", "fresher-strong", "fresher-possible", "fresher-weak",
        "mid-vs-mid", "no-level"])
def test_fit_rules(role, years, you, met, total, verdict):
    assert assess_fit(role, years, you, met, total).verdict == verdict


def test_fresher_advice_says_what_to_apply_for():
    fit = assess_fit("senior", 5, 0, 9, 10)
    assert "no full-time work experience" in fit.advice and "junior or associate" in fit.advice
    assert "5+ years" in " ".join(fit.reasons)


def setup(summary, years):
    catalog = load_catalog()
    matcher = load_matcher(catalog.skills)
    llm = fake_llm()
    llm.extractions[2] = AdviceDraft(summary=summary, projects=[])
    senior_jd = JD + "\nRequirements: 6+ years of experience building ML systems."
    prep = prepare(llm, RESUME, senior_jd, catalog, matcher, load_guide(), "Senior ML Engineer", years=years)
    return assemble(prep, *advise(llm, prep, matcher)), llm


def test_llm_cannot_call_a_fresher_a_fit_for_a_senior_role():
    out, llm = setup("You are a strong fit for this senior role.", years=0)
    assert out.fit.verdict == "Not yet"
    assert "strong fit" not in out.summary.lower() and out.summary.startswith("Fit for this role: Not yet")
    assert any("called you a fit" in d for d in out.dropped)
    assert "Overall fit (computed from the facts, do not contradict): Not yet" in llm.seen_prompts[2]


def test_honest_summary_is_kept():
    out, _ = setup("Not ready for this senior role yet; apply to junior roles. Good Python base.", years=0)
    assert out.summary.startswith("Not ready for this senior role yet")


def test_questions_are_the_same_every_run():
    jobs = ["Python", "Docker"]
    assert line_questions(RESUME, jobs) == line_questions(RESUME, jobs)


def test_vague_and_self_rated_lines_are_asked_about_first():
    resume = ("Proficient in PyTorch, MLflow, Airflow, Docker, and GCP.\n"
              "Owned CI/CD for automated retraining and PyTest testing\n"
              "Built a recommender in Python using Pandas for a course project")
    qs = line_questions(resume, ["Docker", "Python"])
    assert [q.issue.split("'")[1] for q in qs[:2]] == ["Proficient in", "Owned"]
    assert all(q.resume_line in resume for q in qs)


def test_every_llm_call_has_a_persona():
    out, llm = setup("Good Python base.", years=0)
    assert llm.seen_systems == [prompts.READER, prompts.READER, prompts.ADVISOR]


# ---------- Internships and projects, when the JD says they count ----------

from app.services.experience import ExperienceRules, experience_rules  # noqa: E402


@pytest.mark.parametrize("text, internships, projects, asks", [
    ("1+ year of experience (internships count)", True, False, False),
    ("0-2 years of experience, including internships", True, False, False),
    ("Internship experience is preferred.", False, False, True),
    ("Academic projects will be considered as experience.", False, True, False),
    ("Completed at least one internship in data science", False, False, True),
    ("Experience: 2 years in Python", False, False, False),
    ("Our interns build great things", False, False, False),
], ids=["count-brackets", "including", "preferred", "projects", "at-least-one", "plain-years", "not-a-rule"])
def test_experience_rules_are_read_from_the_jd(text, internships, projects, asks):
    r = experience_rules(text)
    assert (r.internships_count, r.projects_count, r.asks_internship) == (internships, projects, asks)


def test_internships_count_only_when_the_jd_says_so():
    counts = ExperienceRules(internships_count=True)
    # a 2+ year (mid) role: 18 months of internship + 1 year of work = 2.5 years -> same level
    assert assess_fit("mid", 2, 1, 9, 10, internship_months=18, rules=counts).verdict == "Strong fit"
    assert assess_fit("mid", 2, 1, 9, 10, internship_months=18).verdict == "Stretch"   # JD silent: not counted
    assert "internships as experience" in " ".join(
        assess_fit("mid", 2, 1, 9, 10, internship_months=18, rules=counts).reasons)


def test_projects_count_brings_you_one_level_closer():
    assert assess_fit("mid", 2, 0, 9, 10, rules=ExperienceRules(projects_count=True)).verdict == "Strong fit"
    assert assess_fit("senior", 5, 0, 9, 10, rules=ExperienceRules(projects_count=True)).verdict == "Stretch"


def test_missing_required_internship_is_flagged():
    asks = ExperienceRules(asks_internship=True)
    fit = assess_fit("fresher", None, 0, 9, 10, rules=asks)
    assert fit.verdict == "Possible fit" and "look for one" in fit.advice
    assert assess_fit("fresher", None, 0, 9, 10, internship_months=6, rules=asks).verdict == "Strong fit"


def test_senior_candidate_for_a_junior_role_is_never_called_not_yet():
    """The bug: a senior ML engineer applying to a junior analyst job got a level-mismatch verdict."""
    fit = assess_fit("fresher", 0, 8, 9, 10)
    assert fit.verdict == "Strong fit" and fit.overqualified
    assert "more experienced" in fit.advice and "why you want a role at this level" in fit.advice
    weak = assess_fit("fresher", 0, 8, 2, 10)
    assert weak.verdict == "Stretch" and "most must-have skills are missing" in weak.advice   # skills decide, not level
    assert not assess_fit("fresher", 0, 0, 9, 10).overqualified
