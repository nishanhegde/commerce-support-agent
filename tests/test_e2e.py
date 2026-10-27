"""Day 50 tests: the end-to-end scenario replayed offline, and one regression
test per fix in failures.md. No API calls."""

import itertools
from datetime import date

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from support_agent import returns
from support_agent.audit import AuditLog
from support_agent.e2e import judge, run_scenario
from support_agent.hitl import decision
from support_agent.persistence import thread, turn_input
from support_agent.policy import default_policy
from support_agent.react import build_explicit
from support_agent.returns import create_return, eligibility
from support_agent.tools import ORDERS
from test_memory import RecordingModel
from test_react import call, scripted

LATE_ORDER = {"order_id": "1048", "sku": "RS-4471-BLK", "quantity": 1, "reason": "9 days late"}


@pytest.fixture(autouse=True)
def pinned(monkeypatch):
    monkeypatch.setattr(returns, "today", lambda: date(2026, 10, 27))
    # Return ids come from a process-wide counter; restart it so the
    # scripted replies below can name RMA-1048-1.
    monkeypatch.setattr(returns, "_return_seq", itertools.count(1))


def first_turn() -> list[AIMessage]:
    return [
        AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1048"}, id="s1")]),
        AIMessage("", tool_calls=[call("check_return_eligibility", {"order_id": "1048", "sku": "RS-4471-BLK"}, id="e1")]),
        AIMessage("I'm sorry: order 1048 is 9 days late. I can refund ₹7,999 now. Shall I?"),
    ]


# --- the scenario, replayed ------------------------------------------------


def test_scripted_happy_path_passes_every_stage():
    model = scripted(
        *first_turn(),
        AIMessage("", tool_calls=[call("create_return", LATE_ORDER, id="r1")]),
        AIMessage("Done: return RMA-1048-1, ₹7,999 back once the carrier confirms the recall."),
    )
    run = run_scenario(model, approve=True)

    assert [(stage, ok) for stage, ok, _ in judge(run)] == [
        ("triage", True), ("order agent", True), ("eligibility", True), ("approval", True), ("resolution", True),
    ]


def test_scripted_rejection_handled_properly_passes():
    model = scripted(
        *first_turn(),
        AIMessage("", tool_calls=[call("create_return", LATE_ORDER, id="r1")]),
        AIMessage("I'm sorry, the refund was not approved. The order is still on its way."),
    )
    run = run_scenario(model, approve=False)

    assert all(ok for _, ok, _ in judge(run, approved=False))


def test_judge_catches_escalating_around_a_rejection():
    # Live run 3's behaviour (failures.md F6), replayed: the judge must fail it.
    model = scripted(
        *first_turn(),
        AIMessage("", tool_calls=[call("create_return", LATE_ORDER, id="r1")]),
        AIMessage("", tool_calls=[call("escalate_to_human", {"summary": "s", "reason": "get refund actioned"}, id="x1")]),
        AIMessage("Ticket ESC-0001: a person will get the refund actioned."),
    )
    run = run_scenario(model, approve=False)

    stages = {stage: ok for stage, ok, _ in judge(run, approved=False)}
    assert stages["resolution"] is False


def test_judge_catches_a_reply_that_never_says_late():
    model = scripted(
        AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1048"}, id="s1")]),
        AIMessage("Good news! I can help with that."),
        AIMessage("Okay."),
    )
    run = run_scenario(model, approve=True)

    stages = {stage: ok for stage, ok, _ in judge(run)}
    assert stages["triage"] is True and stages["order agent"] is False


# --- F1: delayed shipments can be refunded ---------------------------------


def test_f1_late_shipment_is_refundable_without_delivery():
    result = eligibility(ORDERS["1048"], "RS-4471-BLK", 1, date(2026, 10, 27))

    assert result["eligible"] is True and result["delayed_shipment"] is True
    assert result["refund_amount"] == 7999
    assert "return_by" not in result  # nothing to send back


def test_f1_late_means_more_than_seven_days():
    # ETA 2026-10-18: 7 days late on 10-25 isn't enough, 8 on 10-26 is.
    assert eligibility(ORDERS["1048"], "RS-4471-BLK", 1, date(2026, 10, 25))["eligible"] is False
    assert eligibility(ORDERS["1048"], "RS-4471-BLK", 1, date(2026, 10, 26))["eligible"] is True


def test_f1_on_time_shipment_still_waits():
    result = eligibility(ORDERS["1042"], "RS-4471-BLK", 1, date(2026, 10, 27))  # ETA tomorrow

    assert result["eligible"] is False
    assert "hasn't been delivered yet" in result["reasons"][0]


def test_f1_delayed_refund_recalls_the_parcel():
    result = create_return.invoke(LATE_ORDER)

    assert result["status"] == "carrier_recall"
    assert "refuse delivery" in result["next_step"]


def test_f1_delayed_refund_still_respects_quantity():
    create_return.invoke(LATE_ORDER)

    assert eligibility(ORDERS["1048"], "RS-4471-BLK", 1, date(2026, 10, 27))["eligible"] is False


def test_f1_delayed_refund_over_threshold_needs_a_human():
    v = default_policy(call("create_return", LATE_ORDER))

    assert (v.human, v.rule) == (True, "refund_over_threshold")


# --- F2: impossible cancellations aren't put to a reviewer -------------------


@pytest.mark.parametrize("order_id", ["1048", "1043", "9999"])
def test_f2_cancel_of_a_non_pending_order_goes_through_unreviewed(order_id):
    v = default_policy(call("cancel_order", {"order_id": order_id, "reason": "late"}))

    assert (v.human, v.rule) == (False, "cancel_ineligible")


def test_f2_cancel_of_a_pending_order_is_still_reviewed():
    assert default_policy(call("cancel_order", {"order_id": "1044", "reason": "x"})).human is True


# --- F4: the model knows today's date ----------------------------------------


def test_f4_system_prompt_carries_todays_date():
    model = RecordingModel(messages=iter([AIMessage("Hi.")]))
    build_explicit(model, checkpointer=InMemorySaver()).invoke(turn_input("hi"), thread("t"))

    assert "Today is 2026-10-27." in model.prompts[0][0].content


# --- F6: a rejection is final ------------------------------------------------


def test_f6_rejection_tells_the_model_not_to_route_around_it():
    graph = build_explicit(
        scripted(AIMessage("", tool_calls=[call("create_return", LATE_ORDER, id="r1")]), AIMessage("Sorry.")),
        checkpointer=InMemorySaver(),
        audit=AuditLog(),
    )
    graph.invoke(turn_input("refund 1048"), thread("t"))
    out = graph.invoke(decision(False, "carrier says it's out for delivery", reviewer="priya"), thread("t"))

    [tm] = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert "This decision stands" in tm.content
    assert "don't escalate to get it reversed" in tm.content
