"""A small, bounded tool-calling loop (ReAct style), used by the follow-up chat.

  1. Send the conversation + tool descriptions to the LLM.
  2. If it asks for tools, run them and add the results to the conversation.
  3. Repeat until it answers in plain text, or the step limit is hit.
Tools are an explicit allowlist; bad tool names or arguments come back as messages.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from app.services.llm import LLM


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[[Any, Any], dict]

    def openai_spec(self) -> dict:
        """The tool description in the common JSON format LLM providers accept."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": _clean(self.args_model.model_json_schema())}}


def _clean(schema: Any) -> Any:
    """Drop keys some providers reject (titles, defaults). Inside "properties" the keys are
    argument NAMES (one may be called "title"), so those are kept."""
    if isinstance(schema, list):
        return [_clean(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    cleaned = {}
    for key, value in schema.items():
        if key == "properties":
            cleaned[key] = {name: _clean(prop) for name, prop in value.items()}
        elif key not in ("title", "default"):
            cleaned[key] = _clean(value)
    return cleaned


def call_tool(tools: dict[str, Tool], context: Any, name: str, args: dict) -> dict:
    """Run one tool safely. Bad names and bad arguments come back as messages, not crashes."""
    tool = tools.get(name)
    if tool is None:
        return {"error": f"Unknown tool '{name}'. Available tools: {sorted(tools)}"}
    try:
        parsed = tool.args_model.model_validate(args)
    except ValidationError as error:
        return {"error": f"Invalid arguments for {name}: {error.errors(include_url=False)}"}
    try:
        return tool.run(context, parsed)
    except Exception as error:  # noqa: BLE001 - a bug in one tool must not kill the conversation
        return {"error": f"{name} failed: {error}"}


class Step(BaseModel):
    tool: str
    args: dict
    result: dict


def run_loop(llm: LLM, messages: list[dict], tools: dict[str, Tool], context: Any,
             max_steps: int = 6) -> tuple[str, list[Step], str]:
    """Adds to `messages` in place. Returns (answer, steps, stopped)."""
    steps: list[Step] = []
    for _ in range(max_steps):
        turn = llm.chat(messages, list(tools.values()))
        messages.append({"role": "assistant", "content": turn.text, "raw": turn.raw,
                         "tool_calls": [c.model_dump() for c in turn.tool_calls]})
        if not turn.tool_calls:
            return turn.text, steps, "answered"
        for call in turn.tool_calls:
            result = call_tool(tools, context, call.name, call.args)
            steps.append(Step(tool=call.name, args=call.args, result=result))
            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                             "content": json.dumps(result)})
    return "I stopped before finishing (step limit reached).", steps, "step_limit"
