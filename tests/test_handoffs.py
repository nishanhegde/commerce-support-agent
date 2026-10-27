"""Day 52 tests: direct worker-to-worker handoffs, the context they carry,
their limits, and their trace. Scripted models, no API calls."""

from datetime import date

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from support_agent import returns
from support_agent.audit import AuditLog, format_row
from support_agent.hitl import decision, pending_approval
from support_agent.persistence import thread
from support_agent.returns import RETURNS
from support_agent.supervisor import HANDOFFS, WORKERS, build_supervisor, render, turn_input
from test_memory import RecordingModel
from test_react import call, scripted
from test_supervisor import finish, route, steps

LATE_ORDER = {"order_id": "1048", "sku": "RS-4471-BLK", "quantity": 1, "reason": "9 days late"}
CONTEXT = "Order 1048, 1 x RS-4471-BLK (Trail Runner 4, ₹7,999), BlueDart, ETA 2026-10-18: 9 days late. Customer wants a refund and has confirmed."


@pytest.fixture(autouse=True)
def pinned_today(monkeypatch):
    monkeypatch.setattr(returns, "today", lambda: date(2026, 10, 27))


def transfer(to, reason="customer wants a refund for a late order", context=CONTEXT, id="h1"):
    return AIMessage("", tool_calls=[call(f"transfer_to_{to}", {"reason": reason, "context": context}, id=id)])


def order_agent_hands_off():
    # Looks the order up, then hands off. Exactly two replies: if the run
    # didn't stop at the transfer, the script would run out.
    return scripted(
        AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1048"}, id="s1")]),
        transfer("returns_agent"),
    )


def build(sup, audit=None, **workers):
    return build_supervisor(sup, worker_models=workers, checkpointer=InMemorySaver(), audit=audit or AuditLog())


# --- the handoff -------------------------------------------------------------


def test_order_agent_hands_straight_to_returns_agent():
    sup = RecordingModel(messages=iter([route("order_agent", "Where is order 1048? Customer is angry."),
                                        finish("Refund on its way.")]))
    graph = build(
        sup,
        order_agent=order_agent_hands_off(),
        returns_agent=scripted(AIMessage("Eligible, delayed shipment, ₹7,999. Not opened yet.")),
    )

    assert steps(graph, turn_input("my order is late, refund me"), thread("t")) == [
        "supervisor", "order_agent", "returns_agent", "supervisor",
    ]
    assert len(sup.prompts) == 2  # no supervisor call between the two workers


def test_the_receiver_gets_the_context_not_the_transcript():
    receiver = RecordingModel(messages=iter([AIMessage("Eligible.")]))
    graph = build(
        scripted(route("order_agent", "Where is order 1048?"), finish("ok")),
        order_agent=order_agent_hands_off(),
        returns_agent=receiver,
    )
    graph.invoke(turn_input("refund me"), thread("t"))

    _, brief = receiver.prompts[0]
    assert brief.content == (
        "Handed over to you by the order_agent.\n"
        "Original request: Where is order 1048?\n"
        "Why it's yours: customer wants a refund for a late order\n"
        f"What the order_agent found: {CONTEXT}"
    )
    assert len(receiver.prompts[0]) == 2  # system + brief: none of the order agent's tool traffic


def test_both_reports_reach_the_supervisor():
    sup = RecordingModel(messages=iter([route("order_agent", "1048?"), finish("ok")]))
    graph = build(sup, order_agent=order_agent_hands_off(),
                  returns_agent=scripted(AIMessage("Eligible, ₹7,999.")))
    graph.invoke(turn_input("refund me"), thread("t"))

    final_prompt = sup.prompts[1][0].content
    assert "order_agent was asked: 1048?\n  reported: handed off to returns_agent" in final_prompt
    assert "returns_agent was asked: Handed over to you by the order_agent." in final_prompt


def test_a_handoff_counts_as_a_delegation():
    graph = build(scripted(route("order_agent", "1048?"), finish("ok")),
                  order_agent=order_agent_hands_off(), returns_agent=scripted(AIMessage("Eligible.")))
    graph.invoke(turn_input("refund me"), thread("t"))

    assert graph.get_state(thread("t")).values["delegations"] == 2


# --- limits ------------------------------------------------------------------


def test_ping_pong_stops_at_the_handoff_limit():
    audit = AuditLog()
    graph = build_supervisor(
        scripted(route("order_agent", "1048?"), finish("Let me look into this.")),
        worker_models={
            "order_agent": scripted(transfer("returns_agent", id="h1")),
            "returns_agent": scripted(transfer("order_agent", reason="not shipped?", id="h2")),
        },
        checkpointer=InMemorySaver(),
        audit=audit,
        max_handoffs=1,
    )

    assert steps(graph, turn_input("refund"), thread("t")) == ["supervisor", "order_agent", "returns_agent", "supervisor"]
    outcomes = [(r["detail"]["from"], r["outcome"]) for r in audit.rows("t", "handoff")]
    assert outcomes == [("order_agent", "handed_off"), ("returns_agent", "refused")]
    last_report = graph.get_state(thread("t")).values["reports"][-1]["report"]
    assert "limit is reached" in last_report


def test_handoff_targets_are_other_workers_only():
    for source, targets in HANDOFFS.items():
        assert source in WORKERS and source not in targets
        assert all(t in WORKERS for t in targets)


# --- guard rails after a handoff -------------------------------------------


def test_an_approval_after_a_handoff_still_pauses_and_resumes():
    audit = AuditLog()
    graph = build(
        scripted(route("order_agent", "Customer wants 1048 refunded, confirmed."), finish("Refund approved.")),
        audit=audit,
        order_agent=order_agent_hands_off(),
        returns_agent=scripted(
            AIMessage("", tool_calls=[call("create_return", LATE_ORDER, id="c1")]),
            AIMessage("Opened RMA-1048-1."),
        ),
    )
    graph.invoke(turn_input("refund it"), thread("t"))
    [pending] = pending_approval(graph, thread("t"))
    assert pending["why"] == "refund ₹7,999 is over the ₹5,000 limit"

    graph.invoke(decision(True, reviewer="priya"), thread("t"))

    assert len(RETURNS) == 1
    # Logged once, although order_agent's node ran before the pause and the
    # resume only re-ran returns_agent's.
    assert len(audit.rows("t", "handoff")) == 1


# --- the trace ---------------------------------------------------------------


def test_handoff_is_in_the_audit_log_and_state():
    audit = AuditLog()
    graph = build(scripted(route("order_agent", "1048?"), finish("ok")), audit=audit,
                  order_agent=order_agent_hands_off(), returns_agent=scripted(AIMessage("Eligible.")))
    graph.invoke(turn_input("refund me"), thread("t"))

    [row] = audit.rows("t", "handoff")
    assert row["outcome"] == "handed_off"
    assert row["detail"]["context"] == CONTEXT
    assert "HANDOFF  handed_off    order_agent => returns_agent" in format_row(row)
    [doc] = graph.get_state(thread("t")).values["handoffs"]
    assert (doc["from"], doc["to"]) == ("order_agent", "returns_agent")


def test_transition_rows_record_where_each_worker_went():
    audit = AuditLog()
    graph = build(scripted(route("order_agent", "1048?"), finish("ok")), audit=audit,
                  order_agent=order_agent_hands_off(), returns_agent=scripted(AIMessage("Eligible.")))
    graph.invoke(turn_input("refund me"), thread("t"))

    goto = {r["node"]: r["detail"].get("goto") for r in audit.rows("t", "transition") if r["node"] in WORKERS}
    assert goto == {"order_agent": "returns_agent", "returns_agent": "supervisor"}


def test_render_shows_the_handoff():
    delta = {"handoffs": [{"from": "order_agent", "to": "returns_agent", "reason": "refund"}], "reports": []}

    assert render("order_agent", delta) == ["  order_agent => returns_agent (handoff): refund"]


def test_handoffs_reset_each_turn():
    graph = build(scripted(route("order_agent", "1048?"), finish("ok"), finish("bye")),
                  order_agent=order_agent_hands_off(), returns_agent=scripted(AIMessage("Eligible.")))
    graph.invoke(turn_input("refund me"), thread("t"))
    graph.invoke(turn_input("thanks"), thread("t"))

    assert graph.get_state(thread("t")).values["handoffs"] == []
