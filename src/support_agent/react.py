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

Day 48 adds `load_customer` at the front, on a thread's first turn only, when
the graph has a Store (memory.py):

    START -> load_customer -> agent ...     (first turn: who is this?)
    START -> agent ...                      (later turns: already in state)

Day 49 changes no edges. Who goes through `approve` is now decided by a
policy function (policy.py) instead of a set of names, and every node runs
inside `audited` (audit.py), which logs each transition.

Day 56 adds no edges either. A failing node is retried (RetryPolicy on
`agent`, per-call retry of read-only tools inside `tools`), and when that is
exhausted the node's error_handler (`recover`) ends the run with a reply the
customer can read, or hands the model call to a fallback model. See
resilience.py.
"""

import json
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, ToolException
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import NodeError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from support_agent.audit import AuditLog, audited
from support_agent.memory import CustomerContext, customer_prompt, load_customer, save_customer_note
from support_agent import returns
from support_agent.policy import Policy, default_policy
from support_agent.resilience import (
    Resilience, ToolUnavailable, is_read_only, is_transient, recovery_messages, retry_call,
)
from support_agent.returns import RETURN_TOOLS
from support_agent.tools import TOOLS

MODEL = "claude-opus-5"

SYSTEM = """You are a commerce support agent for an online retailer.

Use your tools to look up facts before answering. Never guess at stock,
prices, or order status. When a question needs several independent lookups,
request them together rather than one at a time. Be brief and concrete.

For a return: check eligibility first and tell the customer the answer and
the reasons. Create the return only once they confirm. If the customer asks
for a person or disputes a policy answer, escalate to a human and give them
the ticket id.

When the customer tells you something that will matter next time (a size, a
contact preference), save it as a customer note."""

# Day 47: the graph's default tools are the order tools plus the return
# tools. The prebuilt agent keeps the order tools: it's there for comparison.
DEFAULT_TOOLS = [*TOOLS, *RETURN_TOOLS, save_customer_note]


def default_model() -> BaseChatModel:
    # Day 56: both off the SDK's defaults, so the failure policy is in one
    # place. The SDK retries 2x on its own (max_retries=2) and waits up to 10
    # minutes (timeout=600), and LangGraph's RetryPolicy on top would multiply
    # that: 3 x 3 attempts, each possibly minutes long. Zero here, so the
    # retries the audit log shows are the only retries there are.
    return ChatAnthropic(model=MODEL, max_tokens=2048, max_retries=0, timeout=60)


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
    # Day 48: the customer, read from the Store by `load_customer` on the
    # first turn. Checkpointed with the rest of the state, so later turns,
    # resumes and restarts all have it without reading the Store again.
    customer: dict


def build_explicit(
    model: BaseChatModel | None = None,
    max_iterations: int = 6,
    checkpointer: BaseCheckpointSaver | None = None,
    # Day 46: the graph no longer owns its tools. The defaults are the
    # in-memory ones; mcp_bridge supplies mcp-commerce's.
    tools: list[BaseTool] = DEFAULT_TOOLS,
    # Day 49: which calls need a human, as a function of the call. Was a set
    # of names; policy.by_name(names) is that set as a policy.
    policy: Policy = default_policy,
    system: str = SYSTEM,
    # Day 48: long-term memory. With a Store, a thread's first turn loads the
    # customer named in the call's context. Without one, the graph is as it was.
    store: BaseStore | None = None,
    # Day 49: where transitions and approvals go. None: an in-memory log, so
    # the graph always has one and no node has to check.
    audit: AuditLog | None = None,
    # Day 51: set when this graph runs as a supervisor's worker, so its audit
    # rows say "returns_agent/tools" rather than a bare "tools".
    name: str | None = None,
    # Day 52: tools after which the run ends without another model call. A
    # worker that has handed off has nothing left to say.
    stop_after: frozenset[str] = frozenset(),
    # Day 56: retry counts, backoff, and the fallback model.
    resilience: Resilience = Resilience(),
):
    audit = audit or AuditLog()
    label = (lambda node: f"{name}/{node}") if name else (lambda node: node)

    # bind_tools attaches the tool schemas to every request, like passing
    # `tools=TOOLS` to client.messages.create in Month 1.
    llm = (model or default_model()).bind_tools(tools)
    fallback_llm = resilience.fallback_model.bind_tools(tools) if resilience.fallback_model else None
    tools_by_name = {t.name: t for t in tools}

    def ask_model(m, state: AgentState) -> dict:
        # The system prompt is prepended per call rather than stored in state,
        # so it never gets duplicated into the history.
        # Day 48: plus the customer block, rebuilt from state each call.
        # Day 50 fix (failures.md F4): plus today's date, or the model can't
        # tell a late order from an on-time one. returns.today, looked up at
        # call time, so tests that pin the date pin it here too.
        prompt = f"{system}\n\nToday is {returns.today().isoformat()}." + customer_prompt(state.get("customer"))
        response = m.invoke([SystemMessage(prompt), *state["messages"]])
        return {"messages": [response], "iterations": state.get("iterations", 0) + 1}

    def agent(state: AgentState) -> dict:
        return ask_model(llm, state)

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
                # Day 50 fix (failures.md F6): say it's final, here, where
                # the model reads it. Without this it escalated to a human to
                # get the rejection overturned.
                content = {
                    "error": f"rejected by a human reviewer: {rejected[call['id']]}",
                    "final": "This decision stands. Tell the customer plainly it was not "
                    "approved. Don't retry it, and don't escalate to get it reversed "
                    "unless the customer asks for a person.",
                }
                status = "error"
            elif tool is None:
                content, status = {"error": f"no such tool: {call['name']}"}, "error"
            else:
                # Day 49: a call that reaches here without a human was let
                # through by a rule. Log that too: "who approved this ₹4,499
                # refund?" needs an answer, and the answer is a rule id.
                verdict = policy(call)
                if verdict is not None and not verdict.human:
                    audit.record(
                        "approval", "auto_approved", node="tools", tool_call_id=call["id"],
                        tool=call["name"], args=call["args"], by="policy", rule=verdict.rule, why=verdict.why,
                    )
                try:
                    # Day 56: a read that couldn't get an answer is asked
                    # again; a write is not, because "no answer" doesn't mean
                    # "didn't happen". One attempt for anything not read-only.
                    content, status = retry_call(
                        lambda: tool.invoke(call["args"]),
                        attempts=resilience.tool_attempts if is_read_only(tool) else 1,
                        base_delay=resilience.base_delay,
                        on_retry=lambda attempt, e: audit.record(
                            "recovery", "retried", node=label("tools"), tool_call_id=call["id"],
                            tool=call["name"], attempt=attempt, error=repr(e),
                        ),
                    ), "success"
                except ToolUnavailable as e:
                    # Tell the model what is true and what to do about it, or
                    # it will call the same dead tool again until
                    # max_iterations. Same device as the rejection above.
                    content = {
                        "error": f"{call['name']} is unavailable: {e}",
                        "final": "The system behind this tool is not responding. Don't call it "
                        "again this turn. Tell the customer plainly that you can't check "
                        "that right now and that they can try again shortly. Don't guess "
                        "at what it would have said.",
                    }
                    status = "error"
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
        #
        # Day 49: the policy decides what's pending, and its reason goes to
        # the reviewer with the call. It runs on both passes (pause and
        # resume), so it must give the same answer both times: it may read,
        # never write.
        last: AIMessage = state["messages"][-1]
        pending = []
        for c in last.tool_calls:
            verdict = policy(c)
            if verdict is not None and verdict.human:
                pending.append({**c, "rule": verdict.rule, "why": verdict.why})
        decision = interrupt({"tool_calls": pending})
        # Only the resume pass gets here, so each decision is logged once.
        approved = bool(decision.get("approved"))
        reason = decision.get("reason") or ("" if approved else "no reason given")
        for c in pending:
            audit.record(
                "approval", "approved" if approved else "rejected", node="approve",
                tool_call_id=c["id"], tool=c["name"], args=c["args"],
                by=decision.get("reviewer", "unknown"), rule=c["rule"], why=c["why"], reason=reason,
            )
        # One decision covers every gated call in this turn. Per-call
        # decisions would be the same shape with a list.
        if approved:
            return {"rejected": {}}
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
        # Day 49: "gated" is now the policy's call, per call and per argument.
        gated = any((v := policy(c)) is not None and v.human for c in calls)
        return "approve" if gated else "tools"

    def route_after_tools(state: AgentState) -> Literal["agent", "give_up", "__end__"]:
        # Day 52: a stop_after tool ends the run here. The last AIMessage is
        # the one whose calls the tools node just ran.
        last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
        if any(c["name"] in stop_after for c in last_ai.tool_calls):
            return END
        # The guard sits *before* the next model call, as in Month 1, so a
        # call that would go over the limit is never sent or paid for.
        return "give_up" if state["iterations"] >= max_iterations else "agent"

    def recover(state: AgentState, error: NodeError) -> Command:
        # Day 56: the graph-level recovery path. Runs when a node raised and
        # its retries (if any) are used up, instead of the exception leaving
        # graph.invoke(). It returns a Command: an update to apply, and where
        # to go next.
        #
        # For the model, one more chance first: the fallback, if there is one,
        # for the failures a second model could fix (an outage or overload,
        # not a bad request).
        cause = error.error
        if error.node == "agent" and fallback_llm is not None and is_transient(cause):
            try:
                update = ask_model(fallback_llm, state)
            except Exception as e:
                audit.record("recovery", "fallback_failed", node=label(error.node), error=repr(e))
            else:
                audit.record("recovery", "fell_back", node=label(error.node), error=repr(cause))
                # Same routing as after a normal `agent` run: the fallback's
                # answer may itself ask for tools, or for approval.
                return Command(update=update, goto=route_after_agent({"messages": update["messages"]}))
        audit.record("recovery", "gave_up", node=label(error.node), error=repr(cause))
        return Command(update={"messages": recovery_messages(state["messages"])}, goto=END)

    def route_start(state: AgentState) -> Literal["load_customer", "agent"]:
        # "At session start" = the thread has no customer in state yet. The
        # checkpoint answers that; there is no separate session object.
        return "agent" if "customer" in state else "load_customer"

    # context_schema declares what `context=` holds, as the state schema
    # declares the state. Nodes receive it through their `runtime` parameter.
    g = StateGraph(AgentState, context_schema=CustomerContext)

    # Day 49: every node is registered wrapped, so every run of it is logged.
    # Routing functions aren't wrapped: they change no state, and where they
    # sent the run is the next row's `node`.
    def add(node: str, fn, **kwargs) -> None:
        g.add_node(node, audited(label(node), fn, audit), **kwargs)

    # Day 56: both nodes that can fail for reasons outside the graph share the
    # recovery path, but only `agent` is retried by the node: see resilience.py
    # for why `tools` retries per call instead.
    add("agent", agent, retry_policy=resilience.node_retry(), error_handler=recover)
    add("tools", run_tools, error_handler=recover)
    add("approve", approve)
    add("give_up", give_up)
    if store is None:
        g.add_edge(START, "agent")
    else:
        add("load_customer", load_customer)
        g.add_conditional_edges(START, route_start)
        g.add_edge("load_customer", "agent")
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
    # Day 48: the Store sits next to the checkpointer. Every node and tool
    # in the run can reach it through the Runtime.
    return g.compile(checkpointer=checkpointer, store=store)


def final_text(result: dict) -> str:
    """The last message's text."""
    return message_text(result["messages"][-1])


def message_text(msg) -> str:
    """A message's text. Claude's content may be a list of blocks."""
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict))
