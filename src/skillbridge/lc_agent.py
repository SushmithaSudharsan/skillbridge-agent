"""M2, version 2: the same agent on LangChain's create_agent.

create_agent runs the loop from agent.py for you (on LangGraph), and response_format
gives the typed Roadmap directly. The tools are the same Python functions.
"""

import json

from skillbridge.agent import SYSTEM_PROMPT
from skillbridge.roadmap import Roadmap, roadmap_problems
from skillbridge.tools import TOOLS, AgentContext, call_tool


def build_langchain_tools(context: AgentContext) -> list:
    from langchain_core.tools import StructuredTool

    def make(tool):
        def run(**kwargs):
            return json.dumps(call_tool(TOOLS, context, tool.name, kwargs))
        return StructuredTool.from_function(func=run, name=tool.name,
                                            description=tool.description,
                                            args_schema=tool.args_model)
    return [make(t) for t in TOOLS.values()]


def run_langchain_agent(model_name: str, context: AgentContext, user_message: str,
                        max_steps: int = 20):
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model

    agent = create_agent(
        model=init_chat_model(model_name, temperature=0),
        tools=build_langchain_tools(context),
        system_prompt=SYSTEM_PROMPT,
        response_format=Roadmap,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]},
                          config={"recursion_limit": max_steps})
    roadmap: Roadmap = result["structured_response"]
    return roadmap, roadmap_problems(roadmap, context), result["messages"]
