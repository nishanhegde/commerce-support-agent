"""Day 44 tests: streamed steps, and the approval gate. No API calls."""

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from support_agent.hitl import decision, pending_approval, render
from support_agent.persistence import sqlite_checkpointer, thread, turn_input
from support_agent.react import build_explicit, final_text
from support_agent.tools import ORDERS
from test_react import call, scripted


def cancel_then(answer: str) -> list[AIMessage]:
    return [
        AIMessage("", tool_calls=[call("cancel_order", {"order_id": "1044", "reason": "wrong size"})]),
        AIMessage(answer),
    ]


def steps(graph, graph_input, config) -> list[str]:
    return [
        node
        for update in graph.stream(graph_input, config, stream_mode="updates")
        for node in update
    ]


def test_stream_yields_each_step_in_order():
    model = scripted(
        AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1042"})]),
        AIMessage("In transit."),
    )
    graph = build_explicit(model, checkpointer=InMemorySaver())

    assert steps(graph, turn_input("where is 1042?"), thread("t")) == ["agent", "tools", "agent"]


def test_lookups_do_not_stop_for_approval():
    model = scripted(
        AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1042"})]),
        AIMessage("In transit."),
    )
    graph = build_explicit(model, checkpointer=InMemorySaver())
    graph.invoke(turn_input("where is 1042?"), thread("t"))

    assert pending_approval(graph, thread("t")) == []


def test_cancel_pauses_before_the_tool_runs():
    graph = build_explicit(scripted(*cancel_then("Cancelled.")), checkpointer=InMemorySaver())

    assert steps(graph, turn_input("cancel 1044"), thread("t")) == ["agent", "__interrupt__"]
    [pending] = pending_approval(graph, thread("t"))
    assert pending["args"] == {"order_id": "1044", "reason": "wrong size"}
    assert ORDERS["1044"]["status"] == "pending"  # nothing has happened yet


def test_approve_runs_the_tool():
    graph = build_explicit(scripted(*cancel_then("Cancelled.")), checkpointer=InMemorySaver())
    graph.invoke(turn_input("cancel 1044"), thread("t"))

    out = graph.invoke(decision(True), thread("t"))

    assert ORDERS["1044"]["status"] == "cancelled"
    assert final_text(out) == "Cancelled."
    assert pending_approval(graph, thread("t")) == []


def test_reject_skips_the_tool_and_tells_the_model():
    graph = build_explicit(scripted(*cancel_then("I couldn't cancel it.")), checkpointer=InMemorySaver())
    graph.invoke(turn_input("cancel 1044"), thread("t"))

    out = graph.invoke(decision(False, "fraud check"), thread("t"))

    assert ORDERS["1044"]["status"] == "pending"
    [tm] = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tm.status == "error"
    assert "fraud check" in json.loads(tm.content)["error"]
    assert final_text(out) == "I couldn't cancel it."


def test_a_lookup_batched_with_a_cancel_runs_once_after_approval():
    # The reason `approve` is its own node: the read and the cancel run
    # together, after the decision, each exactly once.
    model = scripted(
        AIMessage(
            "",
            tool_calls=[
                call("get_order_status", {"order_id": "1044"}, id="read"),
                call("cancel_order", {"order_id": "1044", "reason": "x"}, id="cancel"),
            ],
        ),
        AIMessage("Done."),
    )
    graph = build_explicit(model, checkpointer=InMemorySaver())
    graph.invoke(turn_input("status then cancel 1044"), thread("t"))
    out = graph.invoke(decision(False, "no"), thread("t"))

    results = {m.tool_call_id: m for m in out["messages"] if isinstance(m, ToolMessage)}
    assert results["read"].status == "success"
    assert results["cancel"].status == "error"


def test_a_paused_approval_survives_a_restart(tmp_path):
    db = tmp_path / "cp.sqlite"
    with sqlite_checkpointer(db) as saver:
        graph = build_explicit(scripted(*cancel_then("unused")), checkpointer=saver)
        graph.invoke(turn_input("cancel 1044"), thread("t"))

    # New process: the pause is still there, and resuming finishes the turn.
    with sqlite_checkpointer(db) as saver:
        graph = build_explicit(scripted(AIMessage("Cancelled.")), checkpointer=saver)
        assert pending_approval(graph, thread("t"))
        out = graph.invoke(decision(True), thread("t"))

    assert final_text(out) == "Cancelled."
    assert ORDERS["1044"]["status"] == "cancelled"


def test_without_a_checkpointer_it_stops_but_cannot_resume():
    graph = build_explicit(scripted(*cancel_then("unused")))

    out = graph.invoke(turn_input("cancel 1044"))
    assert "__interrupt__" in out  # still stopped, so still safe
    assert ORDERS["1044"]["status"] == "pending"

    with pytest.raises(RuntimeError, match="without checkpointer"):
        graph.invoke(decision(True))


def test_render():
    ai = AIMessage("", tool_calls=[call("cancel_order", {"order_id": "1044", "reason": "x"})])
    assert list(render("agent", {"messages": [ai]})) == [
        '  agent    -> cancel_order({"order_id": "1044", "reason": "x"})'
    ]
    assert list(render("approve", {"rejected": {"c1": "fraud"}})) == ["  approve  rejected: fraud"]
    assert list(render("approve", {"rejected": {}})) == ["  approve  approved"]
    assert list(render("agent", {"messages": [AIMessage("Hi")]})) == ["agent> Hi"]
    assert list(render("__interrupt__", ())) == []
