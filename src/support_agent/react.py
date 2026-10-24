"""Day 42: the ReAct agent twice: prebuilt, then rebuilt as an explicit graph.

The explicit graph IS Month 1's `agent.run()` loop (commerce-agent-cli's
agent.py), redrawn:

    Month 1 (while loop)                     LangGraph (explicit graph)
    ------------------------------------     -----------------------------------
    messages: list[dict], appended by hand   state["messages"], add_messages reducer
    client.messages.create(...)              `agent` node: model.invoke(messages)
    if stop_reason != "tool_use": break      route_after_agent -> END
    execute(block) for each tool_use         `tools` node -> ToolMessages
    all results in ONE user message          ChatAnthropic merges adjacent ToolMessages
    loop back to the top                     edge tools -> agent
    if iterations >= max_iterations          route_after_tools -> `give_up`

The `while` is gone, and the loop now comes from a cycle in the graph:
agent -> tools -> agent. Nothing else changed.

Two stopping mechanisms, and they are different things:
  - `max_iterations` (ours): a business rule in the graph, ends cleanly with
    an answer the customer can read. Same as Month 1's hit_limit.
  - `recursion_limit` (LangGraph's): a runtime safety net, counts *supersteps*
    (one per node execution here) and raises GraphRecursionError. Every graph
    has one; default is 25. It exists for when your own guard is missing or
    wrong.
"""

import json
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import ToolException
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import ValidationError

from support_agent.tools import TOOLS

MODEL = "claude-opus-5"

SYSTEM = """You are a commerce support agent for an online retailer.

Use your tools to look up facts before answering. Never guess at stock,
prices, or order status. When a question needs several independent lookups,
request them together rather than one at a time. Be brief and concrete."""

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def default_model() -> BaseChatModel:
    return ChatAnthropic(model=MODEL, max_tokens=2048)


# ---------------------------------------------------------------------------
# 1. Prebuilt. One call. Same graph shape as below, built for you.
# ---------------------------------------------------------------------------


def build_prebuilt(model: BaseChatModel | None = None):
    # create_react_agent is deprecated in favour of LangChain 1.x's
    # `langchain.agents.create_agent`, which is the same loop with a middleware
    # layer on top. We use the LangGraph one because it's a plain graph you can
    # print and compare against ours, and it needs no extra package.
    return create_react_agent(model or default_model(), TOOLS, prompt=SYSTEM)


# ---------------------------------------------------------------------------
# 2. Explicit. The same loop, every piece written out.
# ---------------------------------------------------------------------------


class AgentState(MessagesState):
    # MessagesState already declares
    #     messages: Annotated[list[AnyMessage], add_messages]
    # `add_messages` is a *reducer*: when a node returns {"messages": [m]},
    # LangGraph appends m instead of replacing the list. Day 41's keys had no
    # reducer, so they were overwritten, which is right for `intent` but would
    # wipe the conversation here.
    #
    # `iterations` has no reducer, so the latest write wins. It's read with
    # .get(..., 0) because callers start the graph with only `messages`.
    iterations: int


def build_explicit(model: BaseChatModel | None = None, max_iterations: int = 6):
    # bind_tools attaches the tool schemas to every request, like passing
    # `tools=TOOLS` to client.messages.create in Month 1.
    llm = (model or default_model()).bind_tools(TOOLS)

    def agent(state: AgentState) -> dict:
        # The system prompt is prepended per call rather than stored in state,
        # so it never gets duplicated into the history.
        response = llm.invoke([SystemMessage(SYSTEM), *state["messages"]])
        return {"messages": [response], "iterations": state.get("iterations", 0) + 1}

    def tools(state: AgentState) -> dict:
        # Month 1's execute(), per tool call. Errors become ToolMessages with
        # status="error" (sent to Claude as is_error: True) so the model can
        # adapt, instead of an exception killing the run.
        # (langgraph.prebuilt.ToolNode does this for you. Written out here so
        # it's visible.)
        last: AIMessage = state["messages"][-1]
        results = []
        for call in last.tool_calls:
            tool = TOOLS_BY_NAME.get(call["name"])
            if tool is None:
                content, status = {"error": f"no such tool: {call['name']}"}, "error"
            else:
                try:
                    content, status = tool.invoke(call["args"]), "success"
                except (ValidationError, ToolException) as e:
                    # ValidationError is Month 1's TypeError case: arguments the
                    # schema allowed but the function didn't accept.
                    content, status = {"error": f"{call['name']} failed: {e}"}, "error"
            results.append(
                ToolMessage(
                    json.dumps(content),
                    tool_call_id=call["id"],
                    name=call["name"],
                    status=status,
                )
            )
        # Several ToolMessages in a row. ChatAnthropic folds consecutive ones
        # into a single user turn, which keeps parallel tool calling working
        # (the Day 7 rule) without us doing it by hand.
        return {"messages": results}

    def give_up(state: AgentState) -> dict:
        return {
            "messages": [
                AIMessage(f"[stopped after {max_iterations} iterations without a final answer]")
            ]
        }

    def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
        # Month 1: `if response.stop_reason != "tool_use": break`.
        # LangChain normalises stop_reason away. Checking tool_calls is the
        # provider-neutral equivalent (and what prebuilt `tools_condition` does).
        return "tools" if state["messages"][-1].tool_calls else END

    def route_after_tools(state: AgentState) -> Literal["agent", "give_up"]:
        # The guard sits *before* the next model call, as in Month 1, so a
        # call that would go over the limit is never sent or paid for.
        return "give_up" if state["iterations"] >= max_iterations else "agent"

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("give_up", give_up)
    g.add_edge(START, "agent")
    # No path_map this time: the Literal return types tell LangGraph the
    # possible targets, so the drawn graph still shows every branch.
    g.add_conditional_edges("agent", route_after_agent)
    g.add_conditional_edges("tools", route_after_tools)
    g.add_edge("give_up", END)
    return g.compile()


def final_text(result: dict) -> str:
    """The last message's text. Claude's content may be a list of blocks."""
    msg = result["messages"][-1]
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict))
