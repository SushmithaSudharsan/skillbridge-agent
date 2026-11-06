"""The full analysis with a fake LLM, using the real PeopleX JD.

The fake LLM also tries to sneak in invented content; the checks must remove it.
"""

import pytest

from app.services.analysis import AdviceDraft, ProjectIdea, analyze, compare
from app.services.catalog import load_catalog
from app.services.learning import load_guide
from app.services.llm import FakeLLM
from app.services.jd import PostingDraft, PostingSkillDraft
from app.services.samples import get_sample
from app.services.matching import load_matcher
from app.services.resume import ResumeDraft, ResumeSkillDraft

JD = get_sample("jds", "peoplex_ai_engineer").text
RESUME = get_sample("resumes", "aiml_fresher").text


def fake_llm():
    jd = PostingDraft(skills=[
        PostingSkillDraft(skill="Python", level=3, must_have=True, evidence="Strong command of Python"),
        PostingSkillDraft(skill="RAG", level=1, must_have=True, evidence="retrieval-augmented generation (RAG)"),
        PostingSkillDraft(skill="Docker", level=2, must_have=True, evidence="containerization (e.g., Docker)"),
        PostingSkillDraft(skill="Vector Databases", level=2, must_have=True, evidence="vector DBs"),
        PostingSkillDraft(skill="Kubernetes", level=2, must_have=True, evidence="Kubernetes clusters"),  # invented
        PostingSkillDraft(skill="Pandas", level=2, must_have=True, evidence="Pandas, NumPy, scikit-learn"),
    ])
    resume = ResumeDraft(skills=[
        ResumeSkillDraft(skill="Python", level=2, source="project",
                         evidence="built a content-based recommender in Python using Pandas"),
        ResumeSkillDraft(skill="Pandas", level=1, source="listed", evidence="Python, SQL, Pandas, Machine Learning, Statistics"),
        ResumeSkillDraft(skill="Docker", level=2, source="job", evidence="Deployed services with Docker"),  # invented
    ])
    advice = AdviceDraft(
        summary="Good Python base; needs RAG, Docker and vector databases.",
        projects=[
            ProjectIdea(title="Cited Q&A over course notes", build="A RAG app with a vector store.",
                        job_quote="integrate with databases and vector databases", skills=["RAG", "vector DBs"]),
            ProjectIdea(title="Made-up project", build="x", job_quote="5 years of Kubernetes", skills=["RAG"]),
            ProjectIdea(title="Off-topic project", build="x", job_quote="Strong command of Python",
                        skills=["Power BI"]),
        ])
    return FakeLLM(extractions=[jd, resume, advice])


@pytest.fixture(scope="module")
def result():
    catalog = load_catalog()
    llm = fake_llm()
    out = analyze(llm, RESUME, JD, catalog, load_matcher(catalog.skills), load_guide(), "AI Engineer")
    out._llm = llm
    return out


def test_jd_is_required():
    catalog = load_catalog()
    with pytest.raises(ValueError, match="job description is required"):
        analyze(FakeLLM(), RESUME, "   ", catalog, load_matcher(catalog.skills), load_guide())


def test_statuses(result):
    status = {m.skill: m.status for m in result.matches}
    assert status["Python"] == "below"           # Intermediate vs Advanced
    assert status["Docker"] == "missing"         # the invented Docker evidence was rejected
    assert status["Pandas"] == "below"           # listed only
    assert status["RAG"] == "missing"
    assert "Kubernetes" not in status            # invented JD requirement was rejected


def test_learning_is_course_areas_in_order(result):
    texts = [s.text for s in result.learning]
    assert "Take a course in Advanced Python (Advanced)" in texts
    assert texts.index("Take a course in Fundamentals of LLMs (Beginner)") < texts.index(
        "Take a course in Fundamentals of RAG (Beginner)")
    assert not any("C0" in t or "C1" in t or "C2" in t for t in texts)


def test_only_grounded_projects_survive(result):
    assert [p.title for p in result.projects] == ["Cited Q&A over course notes"]
    assert result.projects[0].skills == ["RAG", "Vector Databases"]


def test_questions_are_chosen_by_rules_and_quote_the_resume(result):
    # the sample resume: the recommender line has no result; the other project line has numbers
    assert [q.resume_line for q in result.questions] == [
        "Movie Recommender: built a content-based recommender in Python using Pandas."]
    assert result.questions[0].issue == "No result or scale is stated."


def test_fit_is_computed_and_stated(result):
    # PeopleX reads as mid-level; the fixture gives no years, so the verdict uses skills only
    assert result.fit.verdict in {"Stretch", "Possible fit", "Strong fit", "Not yet"}
    assert "Overall fit" in result._llm.seen_prompts[2]


def test_listed_only_skill_is_flagged_not_stuffed(result):
    flagged = [c for c in result.checks if c["area"] == "Listed but not shown"]
    assert flagged and "Pandas" in flagged[0]["text"]


def test_every_removal_is_recorded(result):
    assert len(result.dropped) >= 3


def test_seniority_comes_from_jd_hints(result):
    assert result.seniority == "mid"


def test_llm_never_sees_personal_details(result):
    prompts = " ".join(result._llm.seen_prompts)
    assert "priya.raman@example.com" not in prompts and "98765" not in prompts
    assert "Priya" not in prompts


def test_compare_keeps_the_strongest_requirement():
    from app.services.jd import PostingSkill
    from app.services.models import Level
    rows = compare([PostingSkill(skill="SQL", level=1, must_have=False, evidence="SQL", in_catalog=True),
                    PostingSkill(skill="SQL", level=2, must_have=True, evidence="strong SQL", in_catalog=True)], [])
    assert len(rows) == 1 and rows[0].needed == Level.INTERMEDIATE and rows[0].must_have
