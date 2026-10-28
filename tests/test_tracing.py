"""Day 57 tests: the run log. Scripted models carry fake usage numbers."""

import json
from datetime import date

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import ToolException, tool
from langgraph.checkpoint.memory import InMemorySaver

from support_agent import returns
from support_agent.hitl import decision, pending_approval
from support_agent.persistence import thread, turn_input
from support_agent.react import build_explicit, final_text
from support_agent.resilience import Resilience
from support_agent.tracing import RunTrace, cost_usd, format_summary
from test_react import call, scripted

USAGE = {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050, "input_token_details": {"cache_read": 800}}


def ai(text="", calls=(), model="claude-opus-5", usage=USAGE):
    return AIMessage(text, tool_calls=list(calls), usage_metadata=usage, response_metadata={"model_name": model})


def traced(graph_model, text="hi", path=None, **kw):
    trace = RunTrace("t", path=path)
    graph = build_explicit(graph_model, checkpointer=InMemorySaver(), resilience=Resilience(base_delay=0), **kw)
    out = graph.invoke(turn_input(text), thread("t", trace))
    return graph, trace, out


def of_kind(trace, kind):
    return [e for e in trace.events if e["kind"] == kind]


def test_cost_bills_cached_tokens_at_the_cache_rate():
    # 200 uncached in, 800 cache reads, 50 out, at Opus prices.
    assert cost_usd("claude-opus-5", 1000, 50, cache_read=800) == pytest.approx((200 * 5 + 800 * 0.5 + 50 * 25) / 1e6)


def test_cost_matches_a_dated_model_id_and_refuses_an_unknown_one():
    assert cost_usd("claude-haiku-4-5-20251001", 1_000_000, 0) == 1.00
    assert cost_usd("some-future-model", 1000, 1000) is None


def test_a_tool_round_trip_logs_every_node_model_call_and_tool():
    _, trace, _ = traced(scripted(ai(calls=[call("get_order_status", {"order_id": "1042"})]), ai("Shipped.")))

    assert [(e["kind"], e["name"]) for e in trace.events] == [
        ("model", "model"), ("node", "agent"),
        ("tool", "get_order_status"), ("node", "tools"),
        ("model", "model"), ("node", "agent"),
    ]
    assert all(e["ms"] >= 0 and e["thread_id"] == "t" for e in trace.events)
    [first, _] = of_kind(trace, "model")
    assert (first["model"], first["input_tokens"], first["output_tokens"], first["cache_read"]) == ("claude-opus-5", 1000, 50, 800)


def test_summary_totals_the_conversation():
    _, trace, _ = traced(scripted(ai(calls=[call("get_order_status", {"order_id": "1042"})]), ai("Shipped.")))
    s = trace.summary()

    assert (s["model_calls"], s["input_tokens"], s["output_tokens"]) == (2, 2000, 100)
    assert s["cost_usd"] == pytest.approx(2 * (200 * 5 + 800 * 0.5 + 50 * 25) / 1e6)
    assert (s["tool_calls"], s["tool_errors"], s["unpriced_calls"]) == (1, 0, 0)
    assert s["nodes"]["agent"]["runs"] == 2 and s["nodes"]["tools"]["runs"] == 1
    assert "2 calls" in format_summary(s) and "$0.0053" in format_summary(s)


def test_a_model_without_a_price_is_counted_not_guessed():
    _, trace, _ = traced(scripted(ai("Hi.", model="some-future-model")))
    s = trace.summary()

    assert (s["model_calls"], s["cost_usd"], s["unpriced_calls"]) == (1, 0, 1)
    assert "unpriced" in format_summary(s)


def test_a_failing_tool_is_logged_as_an_error():
    @tool
    def broken(order_id: str) -> dict:
        """Refuses."""
        raise ToolException("no such order")

    _, trace, out = traced(scripted(ai(calls=[call("broken", {"order_id": "9"})]), ai("Sorry.")), tools=[broken])

    [t] = of_kind(trace, "tool")
    assert t["status"] == "error" and "no such order" in t["error"]
    assert trace.summary()["tool_errors"] == 1
    assert final_text(out) == "Sorry."


def test_every_retry_attempt_shows_up():
    # Two tool attempts, so two tool events: the retry is visible in the trace
    # as well as in the audit log.
    from support_agent.resilience import ToolUnavailable, mark_read_only

    state = {"n": 0}

    @tool
    def lookup(order_id: str) -> dict:
        """Flaky read."""
        state["n"] += 1
        if state["n"] == 1:
            raise ToolUnavailable("down")
        return {"ok": True}

    mark_read_only(lookup)
    _, trace, _ = traced(scripted(ai(calls=[call("lookup", {"order_id": "1"})]), ai("Done.")), tools=[lookup])

    assert [t["status"] for t in of_kind(trace, "tool")] == ["error", "success"]


def test_a_pause_for_approval_is_not_logged_as_a_failure(monkeypatch):
    monkeypatch.setattr(returns, "today", lambda: date(2026, 10, 27))
    args = {"order_id": "1048", "sku": "RS-4471-BLK", "quantity": 1, "reason": "late"}
    graph, trace, _ = traced(
        scripted(ai(calls=[call("create_return", args)]), ai("Return created.")), "return it"
    )
    assert pending_approval(graph, thread("t"))

    graph.invoke(decision(True, reviewer="r"), thread("t", trace))

    assert all(e["ok"] for e in of_kind(trace, "node"))
    assert [e["name"] for e in of_kind(trace, "node")].count("approve") == 2  # paused, then resumed


def test_events_are_appended_to_a_jsonl_file(tmp_path):
    path = tmp_path / "runs" / "t.jsonl"
    _, trace, _ = traced(scripted(ai("Hello.")), path=path)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == trace.events
    assert {r["kind"] for r in rows} == {"model", "node"}


def test_workers_inside_a_supervisor_are_labelled_with_their_name():
    from support_agent.audit import AuditLog
    from support_agent.supervisor import build_supervisor
    from support_agent.supervisor import turn_input as supervised_input
    from test_supervisor import finish, route

    graph = build_supervisor(
        scripted(route("order_agent", "Status of 1042?"), finish("Shipped.")),
        worker_models={
            "order_agent": scripted(
                ai(calls=[call("get_order_status", {"order_id": "1042"})]), ai("In transit.")
            )
        },
        checkpointer=InMemorySaver(),
        audit=AuditLog(),
    )
    trace = RunTrace("t")
    graph.invoke(supervised_input("where's 1042?"), thread("t", trace))

    names = {e["name"] for e in of_kind(trace, "node")}
    assert {"supervisor", "order_agent/agent", "order_agent/tools"} <= names
    # Tokens from the worker's model calls land in the same conversation total.
    assert trace.summary()["input_tokens"] == 2000
