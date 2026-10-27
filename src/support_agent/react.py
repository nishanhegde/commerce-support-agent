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

Day 44 adds one node, `approve`, between `agent` and `tools`, reached only
when the model asks for a tool in `needs_approval`:

    agent -> approve -> tools -> agent      (cancel: pause for a human first)
    agent -> tools -> agent                 (lookups: unchanged)
"""

import json
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, ToolException
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt
from pydantic import ValidationError

from support_agent.returns import RETURN_TOOLS
from support_agent.tools import NEEDS_APPROVAL, TOOLS

MODEL = "claude-opus-5"

SYSTEM = """You are a commerce support agent for an online retailer.

Use your tools to look up facts before answering. Never guess at stock,
prices, or order status. When a question needs several independent lookups,
request them together rather than one at a time. Be brief and concrete.

For a return: check eligibility first and tell the customer the answer and
the reasons. Create the return only once they confirm. If the customer asks
for a person or disputes a policy answer, escalate to a human and give them
the ticket id."""

# Day 47: the graph's default tools are the order tools plus the return
# tools. The prebuilt agent keeps the order tools: it's there for comparison.
DEFAULT_TOOLS = [*TOOLS, *RETURN_TOOLS]


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
    # Day 44: tool_call id -> the reviewer's reason, for calls a human said no
    # to. Written by `approve`, read by `tools`. No reducer, so each approval
    # replaces the last; stale ids are harmless because ids never repeat.
    rejected: dict[str, str]


def build_explicit(
    model: BaseChatModel | None = None,
    max_iterations: int = 6,
    checkpointer: BaseCheckpointSaver | None = None,
    # Day 46: the graph no longer owns its tools. The defaults are the
    # in-memory ones; mcp_bridge supplies mcp-commerce's, with an approval
    # set taken from the server's annotations.
    tools: list[BaseTool] = DEFAULT_TOOLS,
    needs_approval: set[str] = NEEDS_APPROVAL,
    system: str = SYSTEM,
):
    # bind_tools attaches the tool schemas to every request, like passing
    # `tools=TOOLS` to client.messages.create in Month 1.
    llm = (model or default_model()).bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    def agent(state: AgentState) -> dict:
        # The system prompt is prepended per call rather than stored in state,
        # so it never gets duplicated into the history.
        response = llm.invoke([SystemMessage(system), *state["messages"]])
        return {"messages": [response], "iterations": state.get("iterations", 0) + 1}

    def run_tools(state: AgentState) -> dict:
        # Month 1's execute(), per tool call. Errors become ToolMessages with
        # status="error" (sent to Claude as is_error: True) so the model can
        # adapt, instead of an exception killing the run.
        # (langgraph.prebuilt.ToolNode does this for you. Written out here so
        # it's visible.)
        last: AIMessage = state["messages"][-1]
        rejected = state.get("rejected", {})
        results = []
        for call in last.tool_calls:
            tool = tools_by_name.get(call["name"])
            if call["id"] in rejected:
                # Every tool_use still needs a tool_result, or the next API
                # call is invalid. A rejection is that result: the model
                # reads it and tells the customer.
                content = {"error": f"rejected by a human reviewer: {rejected[call['id']]}"}
                status = "error"
            elif tool is None:
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
                    # MCP tools return text that is already JSON; dumping it
                    # again would send Claude a quoted, escaped string.
                    content if isinstance(content, str) else json.dumps(content),
                    tool_call_id=call["id"],
                    name=call["name"],
                    status=status,
                )
            )
        # Several ToolMessages in a row. ChatAnthropic folds consecutive ones
        # into a single user turn, which keeps parallel tool calling working
        # (the Day 7 rule) without us doing it by hand.
        return {"messages": results}

    def approve(state: AgentState) -> dict:
        # Day 44. interrupt() pauses the graph here: the checkpointer saves the
        # state, and invoke()/stream() return to the caller. The caller resumes
        # with Command(resume=decision), and interrupt() then *returns* that
        # decision.
        #
        # The catch: on resume, LangGraph re-runs this node from its first
        # line, not from the interrupt() call. Python can't freeze a function
        # mid-call, so "resume" means "run it again, and this time interrupt()
        # returns instead of pausing". Anything before interrupt() runs twice.
        # That's why approval is its own node with no side effects, rather
        # than an interrupt() inside `tools`, where a read-only tool batched
        # alongside the cancel would run once before the pause and again after.
        last: AIMessage = state["messages"][-1]
        pending = [c for c in last.tool_calls if c["name"] in needs_approval]
        decision = interrupt({"tool_calls": pending})
        # One decision covers every gated call in this turn. Per-call
        # decisions would be the same shape with a list.
        if decision.get("approved"):
            return {"rejected": {}}
        reason = decision.get("reason") or "no reason given"
        return {"rejected": {c["id"]: reason for c in pending}}

    def give_up(state: AgentState) -> dict:
        return {
            "messages": [
                AIMessage(f"[stopped after {max_iterations} iterations without a final answer]")
            ]
        }

    def route_after_agent(state: AgentState) -> Literal["approve", "tools", "__end__"]:
        # Month 1: `if response.stop_reason != "tool_use": break`.
        # LangChain normalises stop_reason away. Checking tool_calls is the
        # provider-neutral equivalent (and what prebuilt `tools_condition` does).
        calls = state["messages"][-1].tool_calls
        if not calls:
            return END
        # Day 44: only turns that touch a gated tool stop for a human. Lookups
        # go straight through.
        return "approve" if any(c["name"] in needs_approval for c in calls) else "tools"

    def route_after_tools(state: AgentState) -> Literal["agent", "give_up"]:
        # The guard sits *before* the next model call, as in Month 1, so a
        # call that would go over the limit is never sent or paid for.
        return "give_up" if state["iterations"] >= max_iterations else "agent"

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", run_tools)
    g.add_node("approve", approve)
    g.add_node("give_up", give_up)
    g.add_edge(START, "agent")
    g.add_edge("approve", "tools")
    # No path_map this time: the Literal return types tell LangGraph the
    # possible targets, so the drawn graph still shows every branch.
    g.add_conditional_edges("agent", route_after_agent)
    g.add_conditional_edges("tools", route_after_tools)
    g.add_edge("give_up", END)
    # Day 43: with a checkpointer, every superstep's state is saved under the
    # caller's thread_id. Without one, state lives only for one invoke().
    # Day 44: interrupt() needs one, since a pause is just a saved checkpoint.
    # Without it, a turn that reaches `approve` still stops, but nothing was
    # saved to come back to: resuming raises.
    return g.compile(checkpointer=checkpointer)


def final_text(result: dict) -> str:
    """The last message's text."""
    return message_text(result["messages"][-1])


def message_text(msg) -> str:
    """A message's text. Claude's content may be a list of blocks."""
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict))
