"""The follow-up chat agent, scripted with a fake LLM."""

import pytest

from app.services.analysis import advise, assemble, prepare
from app.services.catalog import load_catalog
from app.services.chat import CHAT_TOOLS, ChatSession
from app.services.learning import load_guide
from app.services.llm import FakeLLM, ModelTurn, ToolCall
from app.services.matching import load_matcher
from app.services.agent import call_tool
from test_analysis import JD, RESUME, fake_llm


@pytest.fixture
def session_parts():
    catalog = load_catalog()
    matcher = load_matcher(catalog.skills)
    llm = fake_llm()
    prep = prepare(llm, RESUME, JD, catalog, matcher, load_guide(), "AI Engineer")
    return prep, assemble(prep, *advise(llm, prep, matcher))


def call(name, **args):
    return ModelTurn(tool_calls=[ToolCall(id="c1", name=name, args=args)])


def test_answers_with_a_tool(session_parts):
    prep, analysis = session_parts
    llm = FakeLLM(turns=[call("explain_gap", skill="docker"), ModelTurn(text="Docker is missing.")])
    with ChatSession(llm, prep, analysis) as chat:
        assert chat.ask("Why is Docker a gap?") == "Docker is missing."
        result = chat.steps[0].result
        assert result["status"] == "missing" and result["job_says"] == "containerization (e.g., Docker)"


def test_wording_needs_confirmation_first(session_parts):
    prep, analysis = session_parts
    with ChatSession(FakeLLM(), prep, analysis) as chat:
        out = call_tool(CHAT_TOOLS, chat.context, "honest_wording",
                        {"skill": "Docker", "resume_line": "Movie Recommender"})
        assert "record_answer" in out["error"]


def test_record_must_quote_the_student(session_parts):
    prep, analysis = session_parts
    with ChatSession(FakeLLM(), prep, analysis) as chat:
        chat.context.last_user = "I used Docker to package my movie recommender."
        bad = call_tool(CHAT_TOOLS, chat.context, "record_answer",
                        {"skill": "Docker", "used": True, "how": "used_at_work", "student_words": "I ran Docker in production"})
        assert "error" in bad
        good = call_tool(CHAT_TOOLS, chat.context, "record_answer",
                         {"skill": "Docker", "used": True, "how": "used_in_project",
                          "student_words": "used Docker to package my movie recommender"})
        assert good["used"]
        w = call_tool(CHAT_TOOLS, chat.context, "honest_wording",
                      {"skill": "Docker", "resume_line": "built a content-based recommender in Python using Pandas"})
        assert w["claim_at_most"] == "Beginner" and "project line" in w["how_to_word_it"]


def test_not_used_means_no_wording(session_parts):
    prep, analysis = session_parts
    with ChatSession(FakeLLM(), prep, analysis) as chat:
        chat.context.last_user = "No, I have never used Docker."
        call_tool(CHAT_TOOLS, chat.context, "record_answer",
                  {"skill": "Docker", "used": False, "how": "tried", "student_words": "never used Docker"})
        w = call_tool(CHAT_TOOLS, chat.context, "honest_wording", {"skill": "Docker", "resume_line": "x"})
        assert "Don't add Docker" in w["advice"]


def test_prioritise_runs_debate_once(session_parts):
    from app.services.debate import FocusReason, Ranking
    prep, analysis = session_parts
    g = [m.skill for m in prep.gaps]
    llm = FakeLLM(extractions=[Ranking(reasoning="r", order=g)] * 3 + [FocusReason(reason="Basics first.")])
    with ChatSession(llm, prep, analysis) as chat:
        first = call_tool(CHAT_TOOLS, chat.context, "prioritise", {})
        again = call_tool(CHAT_TOOLS, chat.context, "prioritise", {})
        assert first == again and first["focus"] == g[:3]
        assert len(llm.seen_prompts) == 4           # cached: no second debate


def test_memory_is_wiped_on_exit(session_parts):
    prep, analysis = session_parts
    llm = FakeLLM(turns=[ModelTurn(text="Hi.")])
    with ChatSession(llm, prep, analysis) as chat:
        chat.ask("hello")
        assert len(chat.messages) == 3
    assert chat.messages == [] and chat.context is None
    with pytest.raises(RuntimeError, match="ended"):
        chat.ask("still there?")


def test_system_prompt_has_no_personal_details(session_parts):
    prep, analysis = session_parts
    chat = ChatSession(FakeLLM(), prep, analysis)
    assert "priya" not in chat.messages[0]["content"].casefold()
