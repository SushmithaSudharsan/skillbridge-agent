"""Tests for the agent's tools. No LLM involved."""

import pytest

from skillbridge.catalog import load_catalog
from skillbridge.models import StudentSkill
from skillbridge.tools import TOOLS, AgentContext, call_tool


@pytest.fixture(scope="module")
def context():
    skills = [
        StudentSkill(skill="Python", level=2, source="project", evidence="Built X in Python"),
        StudentSkill(skill="LLMs", level=1, source="project", evidence="GPT-4o chatbot"),
    ]
    return AgentContext(catalog=load_catalog(), student_skills=skills)


def call(context, name, **args):
    return call_tool(TOOLS, context, name, args)


def test_every_tool_has_a_useful_description():
    for tool in TOOLS.values():
        assert len(tool.description) > 40, tool.name


def test_tool_specs_are_valid_json_schema(context):
    for tool in TOOLS.values():
        spec = tool.openai_spec()
        assert spec["function"]["name"] == tool.name
        assert spec["function"]["parameters"]["type"] == "object"


def test_tool_specs_have_no_null_types_or_titles():
    """Some providers (e.g. Gemini) reject optional/null types and extra keys."""
    import json
    for tool in TOOLS.values():
        params = tool.openai_spec()["function"]["parameters"]
        text = json.dumps(params)
        assert '"null"' not in text and "anyOf" not in text, tool.name
        assert "title" not in params
        for name, prop in params.get("properties", {}).items():
            assert "title" not in prop and "default" not in prop, (tool.name, name)
        # every argument the model is allowed to pass is still there
        assert set(params.get("properties", {})) == set(tool.args_model.model_fields), tool.name


def test_match_job_title_supported(context):
    result = call(context, "match_job_title", title="Senior LLM Engineer")
    assert result == {"supported": True, "family": "GenAI Engineer", "seniority": "senior",
                      "seniority_source": "title", "note": "Taken from a level word in the job title"}


def test_match_job_title_unsupported(context):
    result = call(context, "match_job_title", title="Frontend Developer")
    assert result["supported"] is False
    assert "GenAI Engineer" in result["supported_families"]


def test_find_gaps_uses_context_skills(context):
    result = call(context, "find_skill_gaps", family="GenAI Engineer", seniority="fresher")
    skills = [g["skill"] for g in result["gaps"]]
    assert "Python" not in skills                    # level 2 already meets fresher
    assert "Prompt Engineering" in skills
    assert "Python" in result["already_meets"]
    assert result["requirements_source"].startswith("curated profile")


def test_plan_returns_courses_in_order(context):
    result = call(context, "plan_learning_path", family="GenAI Engineer", seniority="fresher")
    ids = [c["course_id"] for c in result["courses"]]
    assert ids and result["total_weeks"] > 0
    assert all(c["reason"] for c in result["courses"])


def test_plan_focus_skills_limits_the_plan(context):
    result = call(context, "plan_learning_path", family="GenAI Engineer", seniority="fresher",
                  focus_skills=["Git"])
    assert [c["course_id"] for c in result["courses"]] == ["C007"]


def test_unknown_family_is_an_error_message(context):
    result = call(context, "find_skill_gaps", family="Data Engineer", seniority="mid")
    assert "Unknown family" in result["error"]


def test_bad_arguments_are_an_error_message(context):
    result = call(context, "find_skill_gaps", family="GenAI Engineer", seniority="junior")
    assert "Invalid arguments" in result["error"]


def test_unknown_tool_is_an_error_message(context):
    result = call(context, "delete_everything")
    assert "Unknown tool" in result["error"]


def test_search_courses(context):
    result = call(context, "search_courses", skill="rag", level="Intermediate")
    assert [c["course_id"] for c in result["courses"]] == ["C117"]


def test_search_unknown_skill(context):
    result = call(context, "search_courses", skill="Kubernetes")
    assert result["courses"] == []


def test_projects_for_role_come_first(context):
    result = call(context, "recommend_projects", family="GenAI Engineer", seniority="fresher")
    assert len(result["projects"]) == 3
    assert result["projects"][0]["project_id"] in {"P009", "P010", "P011", "P012"}


def test_student_skills_tool_shows_evidence(context):
    result = call(context, "get_student_skills")
    assert result["skills"][0]["evidence"]
