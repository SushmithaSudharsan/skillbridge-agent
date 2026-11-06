"""Blockers are found in code from the JD's own words; related skills never count as met."""

from app.services.analysis import compare
from app.services.blockers import find_blockers
from app.services.experience import ExperienceRules
from app.services.jd import PostingSkill
from app.services.models import Level, StudentSkill
from app.services.related import peers

JD = ("Data Analyst. Bachelor's degree in a quantitative field is required.\n"
      "You must be authorized to work in India. This is an on-site role in Pune.\n"
      "2+ years of experience with SQL.")


def by_kind(blockers):
    return {b.kind: b for b in blockers}


def test_blockers_quote_the_job_and_check_what_a_resume_can_answer():
    b = by_kind(find_blockers(JD, "B.Tech in Computer Science, 2026", 2, 0.5))
    assert set(b) == {"work_authorisation", "location", "degree", "experience"}
    assert b["degree"].status == "ok" and b["experience"].status == "check"
    assert b["work_authorisation"].status == "check" and "authorized to work" in b["work_authorisation"].jd_quote
    assert "2+ years" in b["experience"].jd_quote


def test_missing_degree_and_enough_experience():
    b = by_kind(find_blockers(JD, "Diploma in Computer Applications", 2, 3))
    assert b["degree"].status == "check" and b["experience"].status == "ok"
    b = by_kind(find_blockers("Master's degree required.", "B.Tech in CSE", None, 0))
    assert b["degree"].status == "check" and "master's degree" in b["degree"].note


def test_internships_count_when_the_job_says_so():
    rules = ExperienceRules(internships_count=True)
    b = by_kind(find_blockers("2+ years of experience, internships count.", "", 2, 0, rules, internship_months=24))
    assert b["experience"].status == "ok"


def test_no_blockers_when_the_job_has_none():
    assert find_blockers("Build dashboards. Python and SQL.", "B.Tech", None, 0) == []


def test_related_skills_are_peers_not_matches():
    assert "TensorFlow" in peers("PyTorch") and "Tableau" in peers("power bi") and peers("Cooking") == []
    job = [PostingSkill(skill="PyTorch", kind="skill", level=Level.INTERMEDIATE, must_have=True,
                        evidence="PyTorch experience", in_catalog=True)]
    rows = compare(job, [StudentSkill(skill="TensorFlow", level=Level.INTERMEDIATE, source="project",
                                      evidence="Trained a CNN in TensorFlow")])
    row = rows[0]
    assert row.status == "missing" and row.related == "TensorFlow" and row.related_level == Level.INTERMEDIATE
    listed = compare(job, [StudentSkill(skill="TensorFlow", level=Level.BEGINNER, source="listed",
                                        evidence="Skills: TensorFlow")])
    assert listed[0].related is None                    # a skills-list mention is not proof of a related skill
