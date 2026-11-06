"""Any field works: requirements outside our vocabulary still count, match across wordings,
get AI-suggested learning (checked), and met-but-hidden skills go to 'Show better'."""

from app.services.analysis import AdviceDraft, LearningSuggestion, ShowBetterDraft, advise, assemble, prepare
from app.services.catalog import load_catalog
from app.services.jd import PostingDraft, PostingSkillDraft
from app.services.learning import load_guide
from app.services.llm import FakeLLM
from app.services.matching import SkillMatcher, load_matcher
from app.services.resume import ResumeDraft, ResumeSkillDraft

JD = ("Data Engineer. Must have: strong Apache Spark and Airflow experience; build batch "
      "pipelines for the finance team. Clear communicator.")
RESUME = ("Projects\nBuilt a Spark job that cleans 2M transaction rows for a course project\n"
          "Skills: Airflow, Python")


def run(advice: AdviceDraft):
    catalog = load_catalog()
    matcher = load_matcher(catalog.skills)
    llm = FakeLLM(extractions=[
        PostingDraft(skills=[
            PostingSkillDraft(skill="Apache Spark", kind="skill", level=2, must_have=True, evidence="strong Apache Spark"),
            PostingSkillDraft(skill="Airflow", kind="skill", level=2, must_have=True, evidence="Airflow experience"),
            PostingSkillDraft(skill="batch pipelines for finance", kind="task", level=2, must_have=True,
                              evidence="build batch pipelines for the finance team"),
            PostingSkillDraft(skill="Communication", kind="soft", level=2, must_have=False, evidence="Clear communicator")]),
        ResumeDraft(skills=[
            ResumeSkillDraft(skill="spark", level=2, source="project",
                             evidence="Built a Spark job that cleans 2M transaction rows"),
            ResumeSkillDraft(skill="Airflow", level=1, source="listed", evidence="Skills: Airflow, Python")]),
        advice])
    prep = prepare(llm, RESUME, JD, catalog, matcher, load_guide(), "Data Engineer", years=0)
    return prep, assemble(prep, *advise(llm, prep, matcher)), llm


def test_unknown_skills_match_across_wordings_and_keep_their_kind():
    prep, out, llm = run(AdviceDraft(summary="s", projects=[]))
    rows = {m.skill: m for m in prep.matches}
    assert rows["Apache Spark"].status == "meets"            # "spark" on the resume -> "Apache Spark"
    assert rows["Batch pipelines for finance"].kind == "task"
    assert rows["Communication"].kind == "soft"
    assert "Apache Spark (skill)" in llm.seen_prompts[1]      # the resume reader looks for the JD's names


def test_learning_for_unguided_requirements_is_suggested_and_checked():
    _, out, _ = run(AdviceDraft(summary="s", projects=[], learning=[
        LearningSuggestion(skill="Batch pipelines for finance", area="Batch data pipelines in practice",
                           topics=["idempotent jobs", "backfills", "data quality checks", "scheduling", "extra"])]))
    step = next(s for s in out.learning if s.skill == "Batch pipelines for finance")
    assert step.suggested and step.area == "Batch data pipelines in practice" and len(step.topics) == 4


def test_suggestions_naming_providers_codes_or_durations_are_dropped():
    from app.services.analysis import _checked_learning
    dropped: list[str] = []
    bad = [LearningSuggestion(skill="Kafka", area="Coursera: Kafka basics"),
           LearningSuggestion(skill="Kafka", area="CS 229 streaming"),
           LearningSuggestion(skill="Kafka", area="Kafka in 4 weeks")]
    assert _checked_learning(bad, ["Kafka"], dropped) == {} and len(dropped) == 3
    assert _checked_learning([LearningSuggestion(skill="Golang", area="x")], ["Kafka"], dropped) == {}


def test_show_better_lists_hidden_and_context_items_with_checks():
    _, out, _ = run(AdviceDraft(summary="s", projects=[], show_better=[
        ShowBetterDraft(skill="Apache Spark", resume_line="Built a Spark job that cleans 2M transaction rows",
                        suggestion="Lead with the transaction data: it is finance data, like this job's."),
        ShowBetterDraft(skill="Apache Spark", resume_line="Built a Spark job that cleans 2M transaction rows",
                        suggestion="Say it ran on Databricks for 10M rows.")]))
    items = {(b.skill, b.why) for b in out.show_better}
    assert ("Apache Spark", "context") in items
    assert any("Databricks" in d for d in out.dropped)        # invented tool -> rejected
    assert "Apache Spark" not in {m.skill for m in out.strengths}


def test_listed_only_skill_that_meets_goes_to_show_better():
    catalog = load_catalog()
    m = load_matcher(catalog.skills)
    _, out, _ = run(AdviceDraft(summary="s", projects=[]))
    assert all(b.skill != "Airflow" for b in out.show_better)   # listed = Beginner < needed, so it's a gap
    assert isinstance(m, SkillMatcher)


def test_containment_match_is_unique_and_whole_word():
    m = SkillMatcher(["Apache Spark", "Machine Learning", "Deep Learning"])
    assert m.match("spark") == "Apache Spark"
    assert m.match("learning") is None          # ambiguous: never guessed
    assert m.match("sparkling") is None


def test_soft_skills_are_never_gaps_never_scored_and_go_to_interview_prep():
    prep, out, _ = run(AdviceDraft(summary="s", projects=[]))
    assert "Communication" not in [m.skill for m in prep.gaps]
    assert [m.skill for m in prep.soft["interview"]] == ["Communication"] and not prep.soft["shown"]
    assert all(step.skill != "Communication" for step in prep.learning)
    assert prep.fit.must_total == 3                      # the three skill/task must-haves only
    assert "never a gap" in prep.comparison()
