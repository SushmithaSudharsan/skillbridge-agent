"""Every LLM call is traced: step, tokens, latency, cost, prompt version. No text is logged."""

import json
import logging
import os

import pytest

from app.core import prompts
from app.core.logging import JsonFormatter
from app.core.tracing import TracedLLM, price_for
from app.services.jd import PostingDraft
from app.services.llm import FakeLLM


def test_records_cost_and_prompt_version():
    seen = []
    llm = TracedLLM(FakeLLM(extractions=[PostingDraft(skills=[])]), "google_genai:gemini-2.5-flash-lite",
                    on_record=seen.append)
    llm.extract(PostingDraft, "x" * 4000, system="y" * 400)
    r = llm.records[0]
    assert r.step == "PostingDraft" and r.input_tokens == 1100 and r.estimated
    assert r.cost_usd == pytest.approx(1100 * 0.10 / 1e6, abs=1e-6)
    assert r.prompt_set == prompts.set_version() and seen == [r]


def test_failed_calls_are_recorded_and_reraised():
    llm = TracedLLM(FakeLLM(), "unknown-model")
    with pytest.raises(Exception):
        llm.extract(PostingDraft, "prompt")
    assert llm.records[0].ok is False and llm.records[0].cost_usd is None


def test_price_override(monkeypatch):
    monkeypatch.setattr(os, "environ", {"SKILLBRIDGE_PRICE_IN": "1", "SKILLBRIDGE_PRICE_OUT": "2"})
    assert price_for("anything") == (1.0, 2.0)


def test_log_line_is_json_without_text():
    record = logging.LogRecord("skillbridge.llm", logging.INFO, "", 0, "llm_call", None, None)
    record.fields = {"event": "llm_call", "input_tokens": 10}
    line = json.loads(JsonFormatter().format(record))
    assert line["event"] == "llm_call" and line["input_tokens"] == 10


def test_every_prompt_file_has_a_version():
    for path in prompts.PROMPTS_DIR.glob("*.md"):
        assert prompts.version(path.stem) >= 1
