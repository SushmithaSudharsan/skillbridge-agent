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
    def extract(self, schema: type[BaseModel], prompt: str) -> BaseModel: ...


# ---------- Fake LLM for tests: replays scripted answers, costs nothing ----------

class FakeLLM:
    """Returns pre-written answers in order. Records every prompt it was given."""

    def __init__(self, turns: list[ModelTurn] | None = None,
                 extractions: list[BaseModel] | None = None):
        self.turns = list(turns or [])
        self.extractions = list(extractions or [])
        self.seen_messages: list[list[dict]] = []
        self.seen_prompts: list[str] = []

    def chat(self, messages: list[dict], tools: list) -> ModelTurn:
        self.seen_messages.append([dict(m) for m in messages])
        if not self.turns:
            raise RuntimeError("FakeLLM ran out of scripted turns")
        return self.turns.pop(0)

    def extract(self, schema: type[BaseModel], prompt: str) -> BaseModel:
        self.seen_prompts.append(prompt)
        if not self.extractions:
            raise RuntimeError("FakeLLM ran out of scripted extractions")
        return self.extractions.pop(0)


# ---------- Real LLM through LangChain (Gemini, OpenAI, Azure, ...) ----------

class LangChainLLM:
    """Wraps any LangChain chat model, e.g. 'google_genai:<model-name>'."""

    def __init__(self, model_name: str, temperature: float = 0):
        from langchain.chat_models import init_chat_model   # imported here so tests don't need LangChain

        self.model_name = model_name
        self.model = init_chat_model(model_name, temperature=temperature)

    def chat(self, messages: list[dict], tools: list) -> ModelTurn:
        model = self.model.bind_tools([t.openai_spec() for t in tools]) if tools else self.model
        reply = model.invoke([_to_langchain(m) for m in messages])
        calls = [
            ToolCall(id=c.get("id") or f"call_{i}", name=c["name"], args=c.get("args") or {})
            for i, c in enumerate(reply.tool_calls or [])
        ]
        return ModelTurn(text=_text_of(reply.content), tool_calls=calls, raw=reply)

    def extract(self, schema: type[BaseModel], prompt: str) -> BaseModel:
        return self.model.with_structured_output(schema).invoke(prompt)


def get_llm() -> LLM:
    """Build the real LLM from environment variables (or a .env file)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    model_name = os.environ.get("SKILLBRIDGE_MODEL")
    if not model_name:
        raise RuntimeError(
            "Set SKILLBRIDGE_MODEL in your .env file, e.g. "
            "SKILLBRIDGE_MODEL=google_genai:<model name from Google AI Studio>"
        )
    return LangChainLLM(model_name)


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
