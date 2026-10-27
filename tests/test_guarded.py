"""Day 49 tests: the refund policy, the approval log, and the audit trail of
every node run. No API calls."""

from datetime import date

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from support_agent import policy, returns
from support_agent.audit import AuditLog
from support_agent.hitl import decision, pending_approval
from support_agent.persistence import thread, turn_input
from support_agent.policy import by_name, default_policy
from support_agent.react import build_explicit
from support_agent.returns import RETURNS
from support_agent.tools import ORDERS
from test_react import call, scripted

ONE_PAIR = {"order_id": "1046", "sku": "RS-3020-BLU", "quantity": 1, "reason": "too small"}  # ₹4,499
TWO_PAIRS = {**ONE_PAIR, "quantity": 2}  # ₹8,998
INSOLES = {"order_id": "1046", "sku": "AC-0031-BLK", "quantity": 1, "reason": "didn't fit"}


@pytest.fixture(autouse=True)
def pinned_today(monkeypatch):
    monkeypatch.setattr(returns, "today", lambda: date(2026, 10, 27))


@pytest.fixture
def audit():
    log = AuditLog()
    yield log
    log.close()


def graph_for(audit, *responses):
    return build_explicit(scripted(*responses), checkpointer=InMemorySaver(), audit=audit)


def returning(args, answer="Done.", id="c1"):
    return [AIMessage("", tool_calls=[call("create_return", args, id=id)]), AIMessage(answer)]


# --- the policy on its own -------------------------------------------------


def test_refund_within_threshold_goes_through():
    v = default_policy(call("create_return", ONE_PAIR))

    assert (v.human, v.rule) == (False, "refund_within_threshold")
    assert "₹4,499" in v.why


def test_refund_over_threshold_needs_a_human():
    v = default_policy(call("create_return", TWO_PAIRS))

    assert (v.human, v.rule) == (True, "refund_over_threshold")
    assert v.why == "refund ₹8,998 is over the ₹5,000 limit"


def test_threshold_is_exclusive(monkeypatch):
    monkeypatch.setattr(policy, "REFUND_APPROVAL_THRESHOLD", 4499)

    assert default_policy(call("create_return", ONE_PAIR)).human is False  # 4,499 is not "above"


def test_ineligible_return_is_not_sent_to_a_human():
    v = default_policy(call("create_return", INSOLES))

    assert (v.human, v.rule) == (False, "return_ineligible")


def test_cancel_is_always_reviewed():
    assert default_policy(call("cancel_order", {"order_id": "1044", "reason": "x"})).human is True


@pytest.mark.parametrize("name", ["get_order_status", "check_return_eligibility", "escalate_to_human"])
def test_lookups_and_tickets_are_not_actions(name):
    assert default_policy(call(name, {})) is None


def test_by_name_policy():
    p = by_name({"create_cart"})

    assert p(call("create_cart", {})).human is True
    assert p(call("get_product", {})) is None


# --- approvals, in the graph -----------------------------------------------


def test_small_refund_runs_without_stopping_and_is_logged_as_auto(audit):
    graph = graph_for(audit, *returning(ONE_PAIR))
    graph.invoke(turn_input("return one pair"), thread("t"))

    assert pending_approval(graph, thread("t")) == []
    assert len(RETURNS) == 1
    [row] = audit.rows("t", "approval")
    assert row["outcome"] == "auto_approved"
    assert row["detail"]["by"] == "policy"
    assert row["detail"]["rule"] == "refund_within_threshold"


def test_large_refund_stops_with_the_reason_for_the_reviewer(audit):
    graph = graph_for(audit, *returning(TWO_PAIRS))
    graph.invoke(turn_input("return both pairs"), thread("t"))

    [pending] = pending_approval(graph, thread("t"))
    assert pending["why"] == "refund ₹8,998 is over the ₹5,000 limit"
    assert RETURNS == {}
    assert audit.rows("t", "approval") == []  # nobody has decided yet


def test_human_approval_is_logged_with_the_reviewer(audit):
    graph = graph_for(audit, *returning(TWO_PAIRS))
    graph.invoke(turn_input("return both pairs"), thread("t"))
    graph.invoke(decision(True, reviewer="priya"), thread("t"))

    [row] = audit.rows("t", "approval")
    assert row["outcome"] == "approved"
    assert row["detail"]["by"] == "priya"
    assert row["detail"]["rule"] == "refund_over_threshold"
    assert len(RETURNS) == 1


def test_rejection_is_logged_with_the_reason(audit):
    graph = graph_for(audit, *returning(TWO_PAIRS, "Sorry."))
    graph.invoke(turn_input("return both pairs"), thread("t"))
    graph.invoke(decision(False, "3 returns this month", reviewer="priya"), thread("t"))

    [row] = audit.rows("t", "approval")
    assert row["outcome"] == "rejected"
    assert row["detail"]["reason"] == "3 returns this month"
    assert RETURNS == {}


def test_mixed_turn_logs_each_call_its_own_way(audit):
    # A small refund and a cancellation in one model turn. The turn stops
    # (for the cancel), but only the cancel is put to the reviewer.
    graph = graph_for(
        audit,
        AIMessage(
            "",
            tool_calls=[
                call("create_return", ONE_PAIR, id="r1"),
                call("cancel_order", {"order_id": "1044", "reason": "changed mind"}, id="k1"),
            ],
        ),
        AIMessage("Both done."),
    )
    graph.invoke(turn_input("return one pair and cancel 1044"), thread("t"))
    assert [c["id"] for c in pending_approval(graph, thread("t"))] == ["k1"]

    graph.invoke(decision(True, reviewer="priya"), thread("t"))

    by_id = {r["detail"]["tool_call_id"]: r for r in audit.rows("t", "approval")}
    assert by_id["k1"]["outcome"] == "approved" and by_id["k1"]["detail"]["by"] == "priya"
    assert by_id["r1"]["outcome"] == "auto_approved"
    assert ORDERS["1044"]["status"] == "cancelled" and len(RETURNS) == 1


# --- the transition trail --------------------------------------------------


def transitions(audit, thread_id):
    return [(r["step"], r["node"], r["outcome"]) for r in audit.rows(thread_id, "transition")]


def test_every_node_run_is_a_row(audit):
    graph = graph_for(
        audit,
        AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1042"})]),
        AIMessage("In transit."),
    )
    graph.invoke(turn_input("where is 1042?"), thread("t"))

    assert transitions(audit, "t") == [
        (1, "agent", "completed"),
        (2, "tools", "completed"),
        (3, "agent", "completed"),
    ]
    first = audit.rows("t", "transition")[0]["detail"]
    assert first["tool_calls"][0]["name"] == "get_order_status"


def test_a_pause_and_its_resume_are_both_visible(audit):
    graph = graph_for(audit, *returning(TWO_PAIRS))
    graph.invoke(turn_input("return both pairs"), thread("t"))
    graph.invoke(decision(True, reviewer="priya"), thread("t"))

    # `approve` ran twice at the same step: Day 44's re-run, on the record.
    assert transitions(audit, "t") == [
        (1, "agent", "completed"),
        (2, "approve", "paused"),
        (2, "approve", "completed"),
        (3, "tools", "completed"),
        (4, "agent", "completed"),
    ]


def test_threads_are_kept_apart(audit):
    graph = graph_for(audit, AIMessage("Hi."), AIMessage("Hello."))
    graph.invoke(turn_input("hi"), thread("a"))
    graph.invoke(turn_input("hi"), thread("b"))

    assert len(audit.rows("a")) == len(audit.rows("b")) == 1


def test_a_failing_node_is_logged_before_the_error_propagates(audit):
    @tool
    def flaky(order_id: str) -> dict:
        """Always breaks."""
        raise RuntimeError("database is down")

    graph = build_explicit(
        scripted(AIMessage("", tool_calls=[call("flaky", {"order_id": "1"})])),
        checkpointer=InMemorySaver(),
        tools=[flaky],
        audit=audit,
    )
    with pytest.raises(RuntimeError):
        graph.invoke(turn_input("x"), thread("t"))

    *_, last = audit.rows("t", "transition")
    assert (last["node"], last["outcome"]) == ("tools", "failed")
    assert "database is down" in last["detail"]["error"]


def test_audit_file_survives_a_restart(tmp_path):
    path = tmp_path / "audit.sqlite"
    with AuditLog(path) as log:
        graph_for(log, AIMessage("Hi.")).invoke(turn_input("hi"), thread("t"))

    with AuditLog(path) as log:
        assert transitions(log, "t") == [(1, "agent", "completed")]
