"""Tests for loading and checking the data files."""

import json

import pytest

from skillbridge.catalog import CatalogError, load_catalog

SENIORITIES = ["fresher", "mid", "senior"]


# ---------- A tiny, valid data set we can break on purpose ----------

def tiny_data() -> dict[str, list]:
    """The smallest data set that passes every check."""
    return {
        "roles.json": [
            {"name": "Data Analyst", "titles": ["Data Analyst", "BI Analyst"]},
        ],
        "role_profiles.json": [
            {"family": "Data Analyst", "seniority": s,
             "requirements": [{"skill": "SQL", "level": 1, "priority": "required"}]}
            for s in SENIORITIES
        ],
        "courses.json": [
            {"course_id": "C1", "course_name": "SQL Basics", "skills_taught": ["SQL"],
             "prerequisites": [], "difficulty": "Beginner", "duration_weeks": 2,
             "careers": ["Data Analyst"]},
        ],
        "portfolio_projects.json": [
            {"project_id": "P1", "project_name": "Sales Report",
             "description": "Query sales data.", "skills": ["SQL"],
             "difficulty": "Beginner", "careers": ["Data Analyst"]},
        ],
    }


def write(folder, data: dict[str, list]) -> None:
    """Write each list in `data` to a JSON file inside `folder`."""
    for filename, rows in data.items():
        (folder / filename).write_text(json.dumps(rows), encoding="utf-8")


# ---------- The real data ----------

def test_real_data_loads():
    catalog = load_catalog()
    assert len(catalog.families) == 5
    assert len(catalog.profiles) == 15


# ---------- The tiny data set ----------

def test_tiny_data_loads(tmp_path):
    write(tmp_path, tiny_data())
    catalog = load_catalog(tmp_path)
    assert catalog.skills == ["SQL"]


def test_course_with_unknown_career_is_rejected(tmp_path):
    data = tiny_data()
    data["courses.json"][0]["careers"] = ["Date Analyst"]
    write(tmp_path, data)
    with pytest.raises(CatalogError, match="unknown career 'Date Analyst'"):
        load_catalog(tmp_path)


def test_duplicate_course_id_is_rejected(tmp_path):
    data = tiny_data()
    data["courses.json"].append(dict(data["courses.json"][0]))
    write(tmp_path, data)
    with pytest.raises(CatalogError, match="Course ID used twice: C1"):
        load_catalog(tmp_path)


def test_missing_profile_is_rejected(tmp_path):
    data = tiny_data()
    data["role_profiles.json"].pop()            # removes "senior"
    write(tmp_path, data)
    with pytest.raises(CatalogError, match="Missing profile: Data Analyst / senior"):
        load_catalog(tmp_path)


def test_unknown_seniority_is_rejected(tmp_path):
    data = tiny_data()
    data["role_profiles.json"][0]["seniority"] = "junior"
    write(tmp_path, data)
    with pytest.raises(CatalogError, match="role_profiles.json has invalid entries"):
        load_catalog(tmp_path)


def test_prerequisite_nobody_teaches_is_rejected(tmp_path):
    data = tiny_data()
    data["courses.json"][0]["prerequisites"] = ["Excel"]
    write(tmp_path, data)
    with pytest.raises(CatalogError, match="needs 'Excel' but no Beginner course"):
        load_catalog(tmp_path)


def test_requirement_without_a_course_for_each_step_is_rejected(tmp_path):
    data = tiny_data()
    data["role_profiles.json"][0]["requirements"][0]["level"] = 2   # needs Intermediate
    write(tmp_path, data)
    with pytest.raises(CatalogError, match="no Intermediate course for SQL"):
        load_catalog(tmp_path)


def test_project_with_unknown_skill_is_rejected(tmp_path):
    data = tiny_data()
    data["portfolio_projects.json"][0]["skills"] = ["Tableau"]
    write(tmp_path, data)
    with pytest.raises(CatalogError, match="P1 uses unknown skill 'Tableau'"):
        load_catalog(tmp_path)


def test_title_shared_by_two_families_is_rejected(tmp_path):
    data = tiny_data()
    data["roles.json"].append({"name": "Data Scientist", "titles": ["bi analyst"]})
    data["role_profiles.json"] += [
        {"family": "Data Scientist", "seniority": s,
         "requirements": [{"skill": "SQL", "level": 1, "priority": "required"}]}
        for s in SENIORITIES
    ]
    write(tmp_path, data)
    with pytest.raises(CatalogError, match="used by more than one family: bi analyst"):
        load_catalog(tmp_path)


def test_missing_file_is_reported(tmp_path):
    data = tiny_data()
    del data["courses.json"]
    write(tmp_path, data)
    with pytest.raises(CatalogError, match="Missing data file"):
        load_catalog(tmp_path)


def test_broken_json_is_reported(tmp_path):
    write(tmp_path, tiny_data())
    (tmp_path / "courses.json").write_text("[ not json", encoding="utf-8")
    with pytest.raises(CatalogError, match="courses.json is not valid JSON"):
        load_catalog(tmp_path)


def test_all_problems_are_reported_together(tmp_path):
    data = tiny_data()
    data["courses.json"][0]["careers"] = ["Date Analyst"]
    data["portfolio_projects.json"][0]["skills"] = ["Tableau"]
    write(tmp_path, data)
    with pytest.raises(CatalogError) as error:
        load_catalog(tmp_path)
    assert "Date Analyst" in str(error.value)
    assert "Tableau" in str(error.value)


# ---------- Looking things up ----------

@pytest.mark.parametrize("title", ["BI Analyst", "bi analyst", "  BI ANALYST  ", "Data Analyst"])
def test_title_lookup_ignores_case_and_spaces(tmp_path, title):
    write(tmp_path, tiny_data())
    catalog = load_catalog(tmp_path)
    assert catalog.family_for_title(title).name == "Data Analyst"


def test_unknown_title_returns_none(tmp_path):
    write(tmp_path, tiny_data())
    catalog = load_catalog(tmp_path)
    assert catalog.family_for_title("Frontend Developer") is None


def test_profile_and_course_lookup(tmp_path):
    write(tmp_path, tiny_data())
    catalog = load_catalog(tmp_path)
    assert catalog.profile("Data Analyst", "mid").seniority == "mid"
    assert catalog.profile("Data Analyst", "lead") is None
    assert catalog.course("C1").course_name == "SQL Basics"
    assert catalog.course("C999") is None


def test_real_titles_map_to_families():
    catalog = load_catalog()
    assert catalog.family_for_title("Machine Learning Engineer").name == "ML Engineer"
    assert catalog.family_for_title("LLM Engineer").name == "GenAI Engineer"
    assert catalog.family_for_title("ML Platform Engineer").name == "MLOps Engineer"
