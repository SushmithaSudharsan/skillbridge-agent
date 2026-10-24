"""Resume reading with a fake LLM: what matters is what Python keeps or rejects."""

from skillbridge.catalog import load_catalog
from skillbridge.llm import FakeLLM
from skillbridge.matching import load_matcher
from skillbridge.resume import ResumeDraft, ResumeSkillDraft, read_resume

RESUME = """Priya Raman | priya.raman@example.com | +91 98765 43210
Skills: Python, SQL, Pandas
Projects: trained a random forest model with scikit-learn on 20,000 listings."""


def draft(*items):
    return ResumeDraft(skills=[ResumeSkillDraft(skill=s, level=l, source=src, evidence=e)
                               for s, l, src, e in items])


def run(llm):
    catalog = load_catalog()
    return read_resume(llm, RESUME, load_matcher(catalog.skills), catalog.skills)


def test_skills_with_real_evidence_are_kept_and_normalised():
    llm = FakeLLM(extractions=[draft(
        ("python", 1, "listed", "Skills: Python, SQL, Pandas"),
        ("sklearn", 2, "project", "trained a random forest model with scikit-learn"),
    )])
    profile = run(llm)
    assert [s.skill for s in profile.skills] == ["Python", "Scikit-learn"]
    assert profile.rejected == []


def test_invented_skill_is_rejected():
    llm = FakeLLM(extractions=[draft(("Kubernetes", 2, "job", "Managed Kubernetes clusters"))])
    profile = run(llm)
    assert profile.skills == []
    assert profile.rejected[0].skill == "Kubernetes"


def test_unknown_but_proven_skill_is_kept_separately():
    llm = FakeLLM(extractions=[draft(("Random Forest", 2, "project", "trained a random forest model"))])
    profile = run(llm)
    assert profile.skills == []
    assert profile.other_skills[0].skill == "Random Forest"


def test_llm_never_sees_email_or_phone():
    llm = FakeLLM(extractions=[draft()])
    run(llm)
    prompt = llm.seen_prompts[0]
    assert "priya.raman@example.com" not in prompt and "98765" not in prompt
    assert "<resume>" in prompt
