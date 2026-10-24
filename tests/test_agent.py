"""Tests for the hand-written agent loop, using a scripted fake LLM."""

import json

from skillbridge.agent import compose_roadmap, run_agent
from skillbridge.catalog import load_catalog
from skillbridge.llm import FakeLLM, ModelTurn, ToolCall
from skillbridge.models import StudentSkill
from skillbridge.roadmap import Roadmap, RoadmapCourse
from skillbridge.tools import AgentContext


def context():
    return AgentContext(catalog=load_catalog(), student_skills=[
        StudentSkill(skill="Python", level=2, source="project", evidence="Python project")])


def call(name, **args):
    return ToolCall(id=f"id_{name}", name=name, args=args)


def test_agent_calls_tools_then_answers():
    llm = FakeLLM(turns=[
        ModelTurn(tool_calls=[call("match_job_title", title="Junior LLM Engineer")]),
        ModelTurn(tool_calls=[call("find_skill_gaps", family="GenAI Engineer", seniority="fresher")]),
        ModelTurn(text="Here is your plan."),
    ])
    run = run_agent(llm, context(), "I want to be a junior LLM engineer")
    assert run.stopped == "answered"
    assert [s.tool for s in run.steps] == ["match_job_title", "find_skill_gaps"]
    assert run.answer == "Here is your plan."


def test_tool_results_go_back_to_the_model_with_the_right_id():
    llm = FakeLLM(turns=[
        ModelTurn(tool_calls=[call("match_job_title", title="Data Analyst")]),
        ModelTurn(text="done"),
    ])
    run_agent(llm, context(), "hi")
    second_call = llm.seen_messages[1]
    tool_message = second_call[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "id_match_job_title"
    assert json.loads(tool_message["content"])["family"] == "Data Analyst"


def test_errors_are_returned_so_the_model_can_fix_them():
    llm = FakeLLM(turns=[
        ModelTurn(tool_calls=[call("find_skill_gaps", family="Data Engineer", seniority="mid")]),
        ModelTurn(text="sorry"),
    ])
    run = run_agent(llm, context(), "hi")
    assert "error" in run.steps[0].result


def test_step_limit_stops_a_looping_model():
    llm = FakeLLM(turns=[ModelTurn(tool_calls=[call("get_student_skills")])] * 5)
    run = run_agent(llm, context(), "hi", max_steps=3)
    assert run.stopped == "step_limit"
    assert len(run.steps) == 3


def test_system_prompt_comes_first():
    llm = FakeLLM(turns=[ModelTurn(text="hello")])
    run_agent(llm, context(), "hi")
    assert llm.seen_messages[0][0]["role"] == "system"
    assert "Never invent" in llm.seen_messages[0][0]["content"]


def test_compose_roadmap_flags_invented_ids():
    invented = Roadmap(target_family="GenAI Engineer", seniority="fresher",
                       requirements_source="curated profile", summary="s",
                       courses=[RoadmapCourse(course_id="C999", why="made up")],
                       project_ids=["P999"], uncovered_gaps=[], next_two_weeks=["start"])
    llm = FakeLLM(turns=[ModelTurn(text="done")], extractions=[invented])
    run = run_agent(llm, context(), "hi")
    roadmap, problems = compose_roadmap(llm, run, context())
    assert "C999 is not in the catalog" in problems
    assert "Project P999 is not in the catalog" in problems
