"""One small interface for talking to an LLM, so the rest of the code never
depends on a particular provider (and tests can use a fake).

Two things an LLM does for us:
  chat(messages, tools)  -> the model's next turn: text, or tool calls
  extract(schema, prompt) -> a filled-in pydantic object (structured output)
"""

import os
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict = Field(default_factory=dict)


class ModelTurn(BaseModel):
    """What the model said back: either a final text answer or tool calls."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    raw: Any = Field(default=None, exclude=True)   # the provider's own reply object, sent back as-is


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list) -> ModelTurn: ...
    def extract(self, schema: type[BaseModel], prompt: str, system: str = "") -> BaseModel: ...


# ---------- Fake LLM for tests: replays scripted answers, costs nothing ----------

class FakeLLM:
    """Returns pre-written answers in order. Records every prompt it was given."""

    def __init__(self, turns: list[ModelTurn] | None = None,
                 extractions: list[BaseModel] | None = None):
        self.turns = list(turns or [])
        self.extractions = list(extractions or [])
        self.seen_messages: list[list[dict]] = []
        self.seen_prompts: list[str] = []
        self.seen_systems: list[str] = []

    def chat(self, messages: list[dict], tools: list) -> ModelTurn:
        self.seen_messages.append([dict(m) for m in messages])
        if not self.turns:
            raise RuntimeError("FakeLLM ran out of scripted turns")
        return self.turns.pop(0)

    def extract(self, schema: type[BaseModel], prompt: str, system: str = "") -> BaseModel:
        self.seen_prompts.append(prompt)
        self.seen_systems.append(system)
        if not self.extractions:
            raise RuntimeError("FakeLLM ran out of scripted extractions")
        return self.extractions.pop(0)


# ---------- Real LLM through LangChain (Gemini, OpenAI, Azure, ...) ----------

class LangChainLLM:
    """Wraps any LangChain chat model, e.g. 'google_genai:<model-name>'."""

    def __init__(self, model_name: str, temperature: float = 0, timeout: float = 60, max_retries: int = 2):
        from langchain.chat_models import init_chat_model   # imported here so tests don't need LangChain

        self.model_name = model_name
        # timeout + retries with backoff are handled by the provider client
        self.model = init_chat_model(model_name, temperature=temperature, timeout=timeout,
                                     max_retries=max_retries)
        self.last_usage: dict[str, int] | None = None     # tokens of the latest call, for tracing

    def chat(self, messages: list[dict], tools: list) -> ModelTurn:
        model = self.model.bind_tools([t.openai_spec() for t in tools]) if tools else self.model
        reply = model.invoke([_to_langchain(m) for m in messages])
        self.last_usage = _usage_of(reply)
        calls = [
            ToolCall(id=c.get("id") or f"call_{i}", name=c["name"], args=c.get("args") or {})
            for i, c in enumerate(reply.tool_calls or [])
        ]
        return ModelTurn(text=_text_of(reply.content), tool_calls=calls, raw=reply)

    def extract(self, schema: type[BaseModel], prompt: str, system: str = "") -> BaseModel:
        """Structured output. `system` (a persona from prompts.py) goes as a system message."""
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        runnable = self.model.with_structured_output(schema, include_raw=True)
        for attempt in (1, 2):                      # invalid structured output: retry once, then fail
            result = runnable.invoke([_to_langchain(m) for m in messages])
            self.last_usage = _usage_of(result.get("raw"))
            if result.get("parsed") is not None:
                return result["parsed"]
        raise ValueError(f"The model returned invalid {schema.__name__} twice: {result.get('parsing_error')}")


def _usage_of(message) -> dict[str, int] | None:
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return None
    return {"input_tokens": int(usage.get("input_tokens", 0)), "output_tokens": int(usage.get("output_tokens", 0))}


def get_llm(judge: bool = False) -> LLM:
    """Build the real LLM from environment variables (or a .env file).
    judge=True reads SKILLBRIDGE_JUDGE_MODEL first: a different model judges more fairly
    (a model tends to prefer its own writing). Falls back to SKILLBRIDGE_MODEL."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    model_name = (judge and os.environ.get("SKILLBRIDGE_JUDGE_MODEL")) or os.environ.get("SKILLBRIDGE_MODEL")
    if not model_name:
        raise RuntimeError(
            "Set SKILLBRIDGE_MODEL in your .env file, e.g. "
            "SKILLBRIDGE_MODEL=google_genai:<model name from Google AI Studio>"
        )
    from app.core.tracing import TracedLLM

    return TracedLLM(LangChainLLM(model_name), model_name)


def _to_langchain(message: dict):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    if message.get("raw") is not None:
        return message["raw"]       # e.g. keeps Gemini's thought signatures intact
    role = message["role"]
    if role == "system":
        return SystemMessage(message["content"])
    if role == "user":
        return HumanMessage(message["content"])
    if role == "assistant":
        calls = [{"name": c["name"], "args": c["args"], "id": c["id"], "type": "tool_call"}
                 for c in message.get("tool_calls", [])]
        return AIMessage(content=message.get("content", ""), tool_calls=calls)
    if role == "tool":
        return ToolMessage(content=message["content"], tool_call_id=message["tool_call_id"],
                           name=message.get("name"))
    raise ValueError(f"Unknown message role: {role}")


def _text_of(content) -> str:
    """Model replies can be a string or a list of parts; return just the text."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts)
