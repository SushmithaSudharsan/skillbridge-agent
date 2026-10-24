"""M2, version 1: the agent loop written by hand, so you can see every step.

  1. Send the conversation + tool descriptions to the LLM.
  2. If it asks for tools, run them and add the results to the conversation.
  3. Repeat until it answers in plain text, or we hit the step limit.
  4. Turn the conversation into a typed Roadmap, then check it in code.
"""

import json

from pydantic import BaseModel

from skillbridge.llm import LLM
from skillbridge.roadmap import Roadmap, roadmap_problems
from skillbridge.tools import TOOLS, AgentContext, call_tool

SYSTEM_PROMPT = """You are SkillBridge, a careful career-learning planner for AI/ML and data roles.

Goal: help the student see their skill gaps for the role they want and give them a learning
roadmap that starts at their current level.

Rules:
- Use the tools for every fact. Never invent courses, projects, skills, or numbers.
- If the student names a job, call match_job_title first.
- If the role is not supported, say so politely and list the supported role families.
- If seniority is unknown, ask the student about their experience instead of guessing.
- Only use course and project IDs that tools returned.
- Never suggest adding skills to a resume that the student does not have.
- When you have what you need, answer in plain text: where they stand, the main gaps,
  and the learning path in the order plan_learning_path returned."""


class Step(BaseModel):
    tool: str
    args: dict
    result: dict


class AgentRun(BaseModel):
    answer: str
    steps: list[Step]
    stopped: str           # "answered" or "step_limit"
    messages: list[dict]


def run_agent(llm: LLM, context: AgentContext, user_message: str, max_steps: int = 8) -> AgentRun:
    tools = list(TOOLS.values())
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}]
    steps: list[Step] = []

    for _ in range(max_steps):
        turn = llm.chat(messages, tools)
        messages.append({"role": "assistant", "content": turn.text, "raw": turn.raw,
                         "tool_calls": [c.model_dump() for c in turn.tool_calls]})
        if not turn.tool_calls:
            return AgentRun(answer=turn.text, steps=steps, stopped="answered", messages=messages)
        for call in turn.tool_calls:
            result = call_tool(TOOLS, context, call.name, call.args)
            steps.append(Step(tool=call.name, args=call.args, result=result))
            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                             "content": json.dumps(result)})

    return AgentRun(answer="I stopped before finishing (step limit reached).", steps=steps,
                    stopped="step_limit", messages=messages)


COMPOSE_PROMPT = """Turn this agent session into a roadmap.
Use only facts from the tool results below. Keep courses in the order plan_learning_path
returned. If a field has no data, leave the list empty.

Tool results:
{results}

Agent's answer:
{answer}"""


def compose_roadmap(llm: LLM, run: AgentRun, context: AgentContext) -> tuple[Roadmap, list[str]]:
    results = "\n".join(f"- {s.tool}({json.dumps(s.args)}) -> {json.dumps(s.result)}"
                        for s in run.steps)
    roadmap: Roadmap = llm.extract(Roadmap, COMPOSE_PROMPT.format(results=results,
                                                                 answer=run.answer))
    return roadmap, roadmap_problems(roadmap, context)
