"""Day 44: watch the agent work, and stop it before it does damage.

Streaming. graph.stream(..., stream_mode="updates") yields one item per node
as each node finishes: {node_name: whatever that node returned}. That's a
step-level view: you see "agent asked for get_order_status", then "tools
returned this", while the turn is still running. invoke() is the same run
with only the final state handed back.

The other modes, for reference: "values" yields the whole state after each
step (more data, same rhythm); "messages" yields LLM tokens as they arrive,
which is what Month 1's streaming did with the raw SDK. Steps are what you
want for watching an agent; tokens are what you want for a chat UI.

Human in the loop. The `approve` node (react.py) calls interrupt(). The
stream then just ends, the pause saved as a checkpoint, and the thread sits
there, through a restart if need be, until someone resumes it with
Command(resume=...). There is no thread blocked waiting and no callback.
"""

import json
from collections.abc import Iterator

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from support_agent.react import message_text


def render(node: str, delta) -> Iterator[str]:
    """Terminal lines for one streamed step.

    `delta` is the dict the node returned, except under the "__interrupt__"
    key, where it's a tuple of Interrupt objects. That one is skipped here:
    `pending_approval` reads the same thing from the checkpoint, which also
    works after a restart, when there was no stream to see it in.
    """
    if node == "__interrupt__" or not delta:
        return
    if node == "approve":
        rejected = delta.get("rejected", {})
        if rejected:
            # One decision per turn, so every rejected call has the same reason.
            yield f"  approve  rejected: {next(iter(rejected.values()))}"
        else:
            yield "  approve  approved"
        return
    for m in delta.get("messages", []):
        if isinstance(m, AIMessage) and m.tool_calls:
            for c in m.tool_calls:
                yield f"  {node:<8} -> {c['name']}({json.dumps(c['args'])})"
        elif isinstance(m, ToolMessage):
            yield f"  {node:<8} <- [{m.status}] {m.content[:100]}"
        elif isinstance(m, AIMessage):
            # The answer itself, in full. Everything above is the working.
            yield f"agent> {message_text(m)}"


def pending_approval(graph, config: dict) -> list[dict]:
    """Tool calls on this thread waiting for a human, or [] if it isn't paused.

    snap.interrupts holds the value each paused interrupt() was called with,
    which `approve` set to {"tool_calls": [...]}.
    """
    snap = graph.get_state(config)
    return [c for i in snap.interrupts for c in i.value["tool_calls"]]


def decision(approved: bool, reason: str = "") -> Command:
    """The input that resumes a paused thread. Whatever goes in resume= is
    what interrupt() returns inside `approve`."""
    return Command(resume={"approved": approved, "reason": reason})
