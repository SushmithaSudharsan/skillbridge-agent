"""Skills your work implies: ML work in a project means you used Python, NumPy and Pandas."""

from app.services.analysis import compare
from app.services.implied import add_implied
from app.services.jd import PostingSkill
from app.services.models import Level, StudentSkill


def skill(name, level, source, evidence="x"):
    return StudentSkill(skill=name, level=Level(level), source=source, evidence=evidence)


def best(skills):
    out = {}
    for s in skills:
        if s.skill not in out or s.level > out[s.skill].level:
            out[s.skill] = s
    return out


def test_ml_project_implies_python_numpy_pandas_at_the_same_level():
    got = best(add_implied([skill("Machine Learning", 2, "project", "built a churn model")]))
    for name in ("Python", "NumPy", "Pandas"):
        assert got[name].level == Level.INTERMEDIATE and got[name].via == "Machine Learning"
        assert got[name].evidence == "built a churn model"


def test_chains_through_two_steps():
    got = best(add_implied([skill("MLOps", 2, "project", "end-to-end MLOps project")]))
    assert got["Machine Learning"].level == Level.INTERMEDIATE
    assert got["NumPy"].via == "MLOps"                      # MLOps -> Machine Learning -> NumPy


def test_capped_at_intermediate_and_never_from_a_skills_list():
    got = best(add_implied([skill("Deep Learning", 3, "job")]))
    assert got["Python"].level == Level.INTERMEDIATE        # Advanced needs its own proof
    assert "Python" not in best(add_implied([skill("Machine Learning", 1, "listed")]))


def test_never_lowers_a_proven_level():
    got = best(add_implied([skill("Python", 3, "job", "led the backend"), skill("Machine Learning", 2, "project")]))
    assert got["Python"].level == Level.ADVANCED and got["Python"].source == "job"


def test_real_use_beats_a_skills_list_on_a_tie_and_shows_why():
    rows = compare([PostingSkill(skill="Python", level=2, must_have=True, evidence="Python", in_catalog=True)],
                   add_implied([skill("Python", 1, "listed", "Python, SQL, R"),
                                skill("Machine Learning", 2, "project", "built a churn model")]))
    assert rows[0].status == "meets"
    assert rows[0].resume_quote == "Implied by your Machine Learning work: built a churn model"
