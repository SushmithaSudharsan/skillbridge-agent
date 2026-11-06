"""The web demo's rules, tested without a web server: access code, samples only, spend cap,
rate limit, in-memory sessions that expire, answers that update the plan, the PDF."""

import pytest

from app.core.config import Settings
from app.core.tracing import TracedLLM
from app.services.analysis import AdviceDraft
from app.services.answers import AnswerClaim, AnswerDraft
from app.services.llm import FakeLLM
from app.services.web import Budget, DemoService, Forbidden, NotFound, SessionStore, TooMany, Unavailable
from test_analysis import JD, fake_llm

SETTINGS = Settings(access_code="open-sesame", daily_budget_usd=1.0, rate_limit_per_hour=3,
                    session_ttl_minutes=30, max_sessions=5, max_jd_chars=15000, trust_proxy=False,
                    enable_docs=False, log_level="WARNING")


def service(llm_script=None, **changes) -> DemoService:
    settings = Settings(**{**SETTINGS.__dict__, **changes})
    script = llm_script or (lambda: fake_llm().extractions)
    return DemoService(settings, lambda: TracedLLM(FakeLLM(extractions=list(script())),
                                                   "google_genai:gemini-2.5-flash-lite"))


def test_access_code_is_required_and_demo_is_locked_without_one():
    svc = service()
    svc.check_access("open-sesame")
    with pytest.raises(Forbidden):
        svc.check_access("wrong")
    with pytest.raises(Forbidden):
        svc.check_access(None)
    with pytest.raises(Unavailable):
        service(access_code=None).check_access("anything")


def test_samples_hide_identity_metadata_and_non_demo_files():
    out = service().samples()
    assert {"aiml_fresher", "ds_fresher_detailed"} <= {r["id"] for r in out["resumes"]}
    assert all("identity" not in r for r in out["resumes"])
    assert "peoplex_ai_engineer" not in {j["id"] for j in out["jds"]}       # marked demo: no


def test_analyze_returns_a_display_view_with_usage():
    view = service().analyze("1.2.3.4", "aiml_fresher", JD, "AI Engineer", years=0)
    assert len(view["session_id"]) == 32
    assert view["fit"]["verdict"] in {"Strong fit", "Possible fit", "Stretch", "Not yet"}
    assert view["counts"]["gaps"] == len(view["learn"]) and view["learning"]
    assert view["usage"]["calls"] == 3 and view["usage"]["cost_usd"] >= 0
    assert all({"question", "answer"} <= set(q) for q in view["questions"])


def test_only_samples_can_be_analysed():
    with pytest.raises(NotFound):
        service().analyze("v", "../../etc/passwd", JD)


def test_rate_limit_per_visitor():
    svc = service()
    for _ in range(3):
        svc.analyze("same-visitor", "aiml_fresher", JD)
    with pytest.raises(TooMany, match="this hour"):
        svc.analyze("same-visitor", "aiml_fresher", JD)
    svc.analyze("other-visitor", "aiml_fresher", JD)


def test_daily_budget_stops_llm_calls():
    svc = service(daily_budget_usd=0.0000001)
    svc.analyze("v", "aiml_fresher", JD)                     # this one spends the budget
    with pytest.raises(TooMany, match="budget"):
        svc.analyze("v2", "aiml_fresher", JD)


def test_answers_update_the_plan_and_pdf_downloads():
    def script():
        return fake_llm().extractions + [
            AnswerDraft(claims=[AnswerClaim(number=1, skill="Docker", level=2, source="project",
                                            evidence="used Docker to package the recommender")]),
            AdviceDraft(summary="Docker is covered now.", projects=[])]
    svc = service(script)
    view = svc.analyze("v", "aiml_fresher", JD, "AI Engineer")
    answers = [""] * len(view["questions"])
    answers[0] = "I used Docker to package the recommender for my final-year project"
    updated = svc.answer("v", view["session_id"], answers)
    assert updated["changes"][0]["skill"] == "Docker"
    assert updated["questions"][0]["answer"].startswith("I used Docker")
    assert svc.report_pdf(view["session_id"]).startswith(b"%PDF")


def test_wrong_number_of_answers_is_rejected():
    svc = service()
    view = svc.analyze("v", "aiml_fresher", JD)
    with pytest.raises(Exception, match="Expected"):
        svc.answer("v", view["session_id"], ["only one"])


def test_end_forgets_the_session():
    svc = service()
    view = svc.analyze("v", "aiml_fresher", JD)
    svc.end(view["session_id"])
    with pytest.raises(NotFound):
        svc.report_pdf(view["session_id"])


def test_sessions_expire_and_are_capped():
    now = [0.0]
    store = SessionStore(ttl_seconds=60, max_items=2, clock=lambda: now[0])
    a = store.put({"x": 1})
    now[0] = 61
    with pytest.raises(NotFound):
        store.get(a)
    store.put({}), store.put({}), store.put({})
    assert len(store) == 2


def test_budget_resets_each_day():
    t = [0.0]
    budget = Budget(daily_usd=0.01, per_hour=100, clock=lambda: t[0])
    from app.core.tracing import CallRecord
    budget.record(CallRecord("s", "m", "p", 0, 0, 0, 0.02, False, True))
    with pytest.raises(TooMany):
        budget.check("v")
    t[0] = 86400 * 2
    budget.check("v")


def test_provider_failure_becomes_a_clear_message():
    svc = DemoService(SETTINGS, lambda: TracedLLM(FakeLLM(), "gemini-2.5-flash-lite"))      # FakeLLM with no script: errors
    with pytest.raises(Unavailable, match="didn't respond"):
        svc.analyze("v", "aiml_fresher", JD)


def test_unconfigured_model_is_reported():
    def no_model():
        raise RuntimeError("Set SKILLBRIDGE_MODEL")
    with pytest.raises(Unavailable, match="isn't configured"):
        DemoService(SETTINGS, no_model).analyze("v", "aiml_fresher", JD)


def test_settings_read_the_env_file(tmp_path, monkeypatch):
    import os
    from app.core.config import get_settings
    (tmp_path / ".env").write_text("DEMO_ACCESS_CODE=from-dotenv\n")
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if k != "DEMO_ACCESS_CODE"})
    try:
        import dotenv  # noqa: F401
    except ImportError:
        return                     # python-dotenv not installed here: nothing to load
    here = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert get_settings().access_code == "from-dotenv"
    finally:
        os.chdir(here)


def test_unpriced_model_stops_the_demo_because_the_spend_cap_could_not_work(monkeypatch):
    monkeypatch.delenv("SKILLBRIDGE_PRICE_IN", raising=False)
    monkeypatch.delenv("SKILLBRIDGE_PRICE_OUT", raising=False)
    svc = DemoService(SETTINGS, lambda: TracedLLM(FakeLLM(extractions=[]), "some-new-model"))
    with pytest.raises(Unavailable, match="no known price"):
        svc.analyze("v", "aiml_fresher", JD)
