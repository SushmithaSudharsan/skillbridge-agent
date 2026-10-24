import pytest

from skillbridge.catalog import load_catalog
from skillbridge.matching import SkillMatcher, load_matcher


@pytest.fixture(scope="module")
def matcher():
    return load_matcher(load_catalog().skills)


@pytest.mark.parametrize("text, expected", [
    ("Python", "Python"), ("python3", "Python"), ("sklearn", "Scikit-learn"),
    ("Scikit Learn", "Scikit-learn"), ("PowerBI", "Power BI"), ("MS Excel", "Excel"),
    ("Retrieval-Augmented Generation", "RAG"), ("agentic workflows", "AI Agents"),
    ("FastAPI", "REST APIs"), ("AWS", "Cloud Computing"), ("Prometheus", "Model Monitoring"),
    ("Dockr", "Docker"),                      # small typo, close spelling match
])
def test_known_names_match(matcher, text, expected):
    assert matcher.match(text) == expected


@pytest.mark.parametrize("text", ["Vector Databases", "Kubernetes", "Java", "Tableau"])
def test_unknown_names_do_not_match(matcher, text):
    assert matcher.match(text) is None


def test_alias_for_unknown_skill_is_rejected():
    with pytest.raises(ValueError):
        SkillMatcher(["Python"], {"Rust": ["rustlang"]})
