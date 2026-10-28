"""Day 58: fifteen scripted conversations, end to end.

Unlike the unit tests next door, each of these is a customer conversation:
one or more turns, the graph's real routing, policy and audit log, and a
scripted model standing in for Claude at every seat. What they assert is what
a customer, a reviewer or an on-call engineer would see: who answered, what
changed, what was logged.

    triage        1-5    which agent takes the question
    approvals     6-10   who allowed the action, and what it changed
    escalations   11-12  a person takes over
    failure paths 13-15  something it depends on is down

Scripts are exact: a model that is asked for one more reply than it was
given runs dry and raises, which is how these catch a run that did extra work.
"""

import itertools
from datetime import date

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from support_agent import escalation, returns
from support_agent.audit import AuditLog
from support_agent.hitl import decision, pending_approval
from support_agent.memory import CustomerContext, seed_profiles
from support_agent.persistence import thread, turn_input as agent_input
from support_agent.react import build_explicit, final_text
from support_agent.resilience import APOLOGY, Resilience
from support_agent.returns import ESCALATIONS, RETURNS
from support_agent.supervisor import build_supervisor, turn_input
from support_agent.tools import ORDERS
from test_escalation import SUMMARY, escalate
from test_resilience import flaky
from test_react import call, scripted
from test_supervisor import finish, refund_graph, route, supervisor_graph

ASHA = CustomerContext("C-100")
LATE_ORDER = {"order_id": "1048", "sku": "RS-4471-BLK", "quantity": 1, "reason": "9 days late"}
FAST = Resilience(base_delay=0)


@pytest.fixture(autouse=True)
def pinned(monkeypatch):
    monkeypatch.setattr(returns, "today", lambda: date(2026, 10, 27))
    monkeypatch.setattr(returns, "_return_seq", itertools.count(1))
    monkeypatch.setattr(escalation, "_ticket_seq", iter(range(1, 100)))


@pytest.fixture
def audit():
    with AuditLog() as log:
        yield log


def nodes_run(audit, tid="t") -> set[str]:
    return {r["node"] for r in audit.rows(tid, "transition")}


def say(graph, text, tid="t", ctx=None):
    return graph.invoke(turn_input(text), thread(tid), context=ctx)


def last_reply(graph, tid="t") -> str:
    return graph.get_state(thread(tid)).values["messages"][-1].content


def recoveries(audit, tid="t") -> list[tuple[str, str]]:
    return [(r["node"], r["outcome"]) for r in audit.rows(tid, "recovery")]


# --- triage: which agent takes the question --------------------------------


def test_01_an_order_question_goes_to_the_order_agent_only(audit):
    graph = supervisor_graph(
        scripted(route("order_agent", "Status of 1042?"), finish("It arrives tomorrow.")),
        audit=audit,
        order_agent=scripted(
            AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1042"})]),
            AIMessage("1042 is in transit, ETA 2026-10-28."),
        ),
    )
    say(graph, "where's order 1042?")

    assert last_reply(graph) == "It arrives tomorrow."
    assert {"order_agent/agent", "order_agent/tools"} <= nodes_run(audit)
    assert not {n for n in nodes_run(audit) if n.startswith(("returns_agent", "general_agent"))}


def test_02_a_refund_question_goes_to_the_returns_agent(audit):
    graph = supervisor_graph(
        scripted(route("returns_agent", "Can 1043 go back?"), finish("The jacket can be returned until 2026-11-19.")),
        audit=audit,
        returns_agent=scripted(
            AIMessage("", tool_calls=[call("check_return_eligibility", {"order_id": "1043", "sku": "JK-2210-GRN"})]),
            AIMessage("Eligible until 2026-11-19."),
        ),
    )
    say(graph, "can I return the jacket from order 1043?")

    assert "returns_agent/tools" in nodes_run(audit)
    assert not any(n.startswith("order_agent") for n in nodes_run(audit))
    assert RETURNS == {}  # asking is not returning


def test_03_a_product_question_goes_to_the_general_agent(audit):
    graph = supervisor_graph(
        scripted(route("general_agent", "Trail shoes under 8000?"), finish("The Trail Runner 4 is ₹7,999.")),
        audit=audit,
        general_agent=scripted(
            AIMessage("", tool_calls=[call("search_products", {"keywords": ["trail"], "max_price": 8000})]),
            AIMessage("Trail Runner 4, ₹7,999, in stock."),
        ),
    )
    say(graph, "any trail shoes under 8000?")

    assert last_reply(graph) == "The Trail Runner 4 is ₹7,999."
    assert "general_agent/tools" in nodes_run(audit)


def test_04_small_talk_needs_no_worker(audit):
    graph = supervisor_graph(scripted(finish("Hi! I can help with orders, returns and products.")), audit=audit)
    say(graph, "hello")

    assert last_reply(graph).startswith("Hi!")
    assert nodes_run(audit) == {"supervisor"}  # no worker, no tool


def test_05_a_two_part_question_uses_two_workers_and_one_reply(audit):
    graph = supervisor_graph(
        scripted(
            route("order_agent", "Status of 1042?", id="r1"),
            route("general_agent", "Do you stock insoles?", id="r2"),
            finish("1042 arrives tomorrow, and yes, Cushion Insoles are in stock at ₹799."),
        ),
        audit=audit,
        order_agent=scripted(
            AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1042"})]),
            AIMessage("In transit, ETA tomorrow."),
        ),
        general_agent=scripted(
            AIMessage("", tool_calls=[call("search_products", {"keywords": ["insole"]})]),
            AIMessage("Cushion Insole, ₹799, in stock."),
        ),
    )
    say(graph, "where's 1042, and do you have insoles?")

    state = graph.get_state(thread("t")).values
    assert [r["worker"] for r in state["reports"]] == ["order_agent", "general_agent"]
    assert state["delegations"] == 2
    assert [m.type for m in state["messages"]] == ["human", "ai"]  # one reply, not two


# --- approvals: who allowed it, and what it changed ------------------------


def test_06_a_large_refund_waits_for_a_person_and_then_happens(audit):
    graph = refund_graph(audit, "Done: your refund is on its way.")
    say(graph, "yes, refund it")

    [pending] = pending_approval(graph, thread("t"))
    assert pending["rule"] == "refund_over_threshold" and RETURNS == {}  # nothing before the decision

    graph.invoke(decision(True, reviewer="priya"), thread("t"))

    [ret] = RETURNS.values()
    assert (ret["order_id"], ret["sku"]) == ("1048", "RS-4471-BLK")
    [row] = audit.rows("t", "approval")
    assert (row["outcome"], row["detail"]["by"], row["detail"]["rule"]) == ("approved", "priya", "refund_over_threshold")


def test_07_a_rejected_refund_changes_nothing_and_the_customer_is_told(audit):
    graph = supervisor_graph(
        scripted(route("returns_agent", "Refund 1048, confirmed."), finish("Sorry, that refund wasn't approved.")),
        audit=audit,
        returns_agent=scripted(
            AIMessage("", tool_calls=[call("create_return", LATE_ORDER, id="c1")]),
            AIMessage("A reviewer rejected it: the carrier shows it out for delivery."),
        ),
    )
    say(graph, "refund it")
    graph.invoke(decision(False, "out for delivery", reviewer="priya"), thread("t"))

    assert RETURNS == {}
    assert last_reply(graph) == "Sorry, that refund wasn't approved."
    [row] = audit.rows("t", "approval")
    assert (row["outcome"], row["detail"]["reason"]) == ("rejected", "out for delivery")


def test_08_a_small_refund_is_approved_by_rule_not_by_a_person(audit):
    # Order 1046: one Road Runner 2 back is ₹4,499, under the ₹5,000 limit.
    args = {"order_id": "1046", "sku": "RS-3020-BLU", "quantity": 1, "reason": "wrong size"}
    graph = build_explicit(
        scripted(AIMessage("", tool_calls=[call("create_return", args)]), AIMessage("Return created.")),
        checkpointer=InMemorySaver(), audit=audit,
    )
    out = graph.invoke(agent_input("wrong size, please return one pair"), thread("t"))

    assert not pending_approval(graph, thread("t"))
    assert final_text(out) == "Return created."
    assert [r["order_id"] for r in RETURNS.values()] == ["1046"]
    [row] = audit.rows("t", "approval")
    assert (row["outcome"], row["detail"]["by"], row["detail"]["rule"]) == ("auto_approved", "policy", "refund_within_threshold")


def test_09_an_ineligible_return_is_refused_without_bothering_a_reviewer(audit):
    # Order 1045 was delivered 2026-09-10: the 30-day window closed 2026-10-10.
    args = {"order_id": "1045", "sku": "RS-3020-BLU", "quantity": 1, "reason": "changed my mind"}
    graph = build_explicit(
        scripted(AIMessage("", tool_calls=[call("create_return", args)]), AIMessage("The return window has closed.")),
        checkpointer=InMemorySaver(), audit=audit,
    )
    out = graph.invoke(agent_input("return my shoes from order 1045"), thread("t"))

    assert not pending_approval(graph, thread("t"))  # nothing to approve: the tool refuses on its own
    assert RETURNS == {}
    assert final_text(out) == "The return window has closed."
    assert [r["detail"]["rule"] for r in audit.rows("t", "approval")] == ["return_ineligible"]


def test_10_a_cancellation_is_rejected_then_approved_on_the_second_ask(audit):
    graph = build_explicit(
        scripted(
            AIMessage("", tool_calls=[call("cancel_order", {"order_id": "1044", "reason": "ordered twice"}, id="c1")]),
            AIMessage("A reviewer declined the cancellation."),
            AIMessage("", tool_calls=[call("cancel_order", {"order_id": "1044", "reason": "ordered twice"}, id="c2")]),
            AIMessage("Order 1044 is cancelled."),
        ),
        checkpointer=InMemorySaver(), audit=audit,
    )
    graph.invoke(agent_input("cancel 1044"), thread("t"))
    graph.invoke(decision(False, "already packed", reviewer="priya"), thread("t"))
    assert ORDERS["1044"]["status"] == "pending"  # a rejection leaves the order alone

    graph.invoke(agent_input("please, I ordered it twice"), thread("t"))
    out = graph.invoke(decision(True, reviewer="priya"), thread("t"))

    assert ORDERS["1044"]["status"] == "cancelled"
    assert final_text(out) == "Order 1044 is cancelled."
    assert [r["outcome"] for r in audit.rows("t", "approval")] == ["rejected", "approved"]


# --- escalations: a person takes over --------------------------------------


def store_with_profiles():
    store = InMemoryStore()
    seed_profiles(store)
    return store


def test_11_a_customer_asks_for_a_manager_and_the_bot_steps_out(tmp_path, audit):
    graph = build_supervisor(
        scripted(route("order_agent", "Status of 1042?"), finish("It arrives tomorrow."), escalate()),
        worker_models={
            "order_agent": scripted(
                AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1042"})]),
                AIMessage("In transit."),
            )
        },
        checkpointer=InMemorySaver(), store=store_with_profiles(), audit=audit, handoff_dir=tmp_path,
        summary_model=scripted(AIMessage("", tool_calls=[call("HandoffSummary", SUMMARY, id="s")])),
    )
    say(graph, "where's 1042?", ctx=ASHA)
    reply = say(graph, "I want to talk to a manager", ctx=ASHA)["messages"][-1].content
    # The supervisor's script is spent. A third model call would raise.
    held = say(graph, "hello? anyone there?", ctx=ASHA)["messages"][-1].content

    assert "HO-0001" in reply and "HO-0001" in held
    assert ESCALATIONS[-1]["after_transfer"] == ["hello? anyone there?"]
    assert [r["outcome"] for r in audit.rows("t", "escalation")] == ["transferred", "message_added"]
    assert "customer: where's 1042?" in (tmp_path / "HO-0001.md").read_text()  # the human sees the whole conversation


def test_12_the_handover_still_happens_when_the_summary_model_is_down(tmp_path, audit):
    graph = build_supervisor(
        scripted(escalate()),
        checkpointer=InMemorySaver(), store=store_with_profiles(), audit=audit, handoff_dir=tmp_path,
        summary_model=flaky(99, AIMessage("never reached")), resilience=FAST,
    )
    reply = say(graph, "get me a person", ctx=ASHA)["messages"][-1].content

    assert "HO-0001" in reply  # the customer is not left waiting on an outage
    assert "AI summary unavailable" in (tmp_path / "HO-0001.md").read_text()
    assert recoveries(audit) == [("escalate", "summary_unavailable")]


# --- failure paths: something it depends on is down -------------------------


def test_13_a_router_outage_ends_the_turn_politely_and_the_thread_recovers(audit):
    # Fails 3 times (all of Resilience.model_attempts), then works again.
    graph = build_supervisor(
        flaky(3, finish("Welcome back, how can I help?")),
        checkpointer=InMemorySaver(), audit=audit, resilience=FAST,
    )
    assert say(graph, "hi")["messages"][-1].content == APOLOGY
    assert recoveries(audit) == [("supervisor", "gave_up")]

    assert say(graph, "hi again")["messages"][-1].content == "Welcome back, how can I help?"


def test_14_a_router_outage_with_a_fallback_model_still_answers(audit):
    graph = build_supervisor(
        flaky(99, finish("never reached")),
        checkpointer=InMemorySaver(), audit=audit,
        resilience=Resilience(base_delay=0, fallback_model=scripted(finish("Backup here: how can I help?"))),
    )
    say(graph, "hi")

    assert last_reply(graph) == "Backup here: how can I help?"
    assert recoveries(audit) == [("supervisor", "fell_back")]


def test_15_a_worker_outage_becomes_a_report_and_the_supervisor_still_replies(audit):
    graph = build_supervisor(
        scripted(route("order_agent", "Status of 1042?"), finish("Sorry, I can't check orders right now. Try again shortly.")),
        worker_models={"order_agent": flaky(99, AIMessage("never reached"))},
        checkpointer=InMemorySaver(), audit=audit, resilience=FAST,
    )
    say(graph, "where's 1042?")

    [report] = graph.get_state(thread("t")).values["reports"]
    assert report["report"] == APOLOGY  # the worker's failure reached the supervisor as an ordinary report
    assert last_reply(graph).startswith("Sorry, I can't check orders")
    assert recoveries(audit) == [("order_agent/agent", "gave_up")]
