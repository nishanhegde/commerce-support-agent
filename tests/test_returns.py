"""Day 47 tests: return eligibility rules, the return tools, and the approval
gate on create_return. No API calls."""

from datetime import date

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from support_agent import returns
from support_agent.hitl import decision, pending_approval
from support_agent.persistence import thread, turn_input
from support_agent.react import build_explicit
from support_agent.returns import (
    ESCALATIONS,
    RETURNS,
    check_return_eligibility,
    create_return,
    eligibility,
    escalate_to_human,
)
from support_agent.tools import ORDERS
from test_react import call, scripted

TODAY = date(2026, 10, 27)


@pytest.fixture(autouse=True)
def pinned_today(monkeypatch):
    # Patches the name the tools call, so "today" is the same on any machine.
    monkeypatch.setattr(returns, "today", lambda: TODAY)


# --- rules -----------------------------------------------------------------


def test_delivered_inside_window_is_eligible():
    result = eligibility(ORDERS["1043"], "JK-2210-GRN", 1, TODAY)

    assert result["eligible"] is True
    assert result["return_by"] == "2026-11-19"  # delivered 10-20 + 30 days
    assert result["refund_amount"] == 5999


def test_window_counts_from_delivery_and_last_day_is_inclusive():
    # 1043 delivered 2026-10-20; 30 days later is 11-19.
    assert eligibility(ORDERS["1043"], "JK-2210-GRN", 1, date(2026, 11, 19))["eligible"] is True
    late = eligibility(ORDERS["1043"], "JK-2210-GRN", 1, date(2026, 11, 20))
    assert late["eligible"] is False
    assert "window closed 2026-11-19" in late["reasons"][0]


def test_outside_standard_window():
    result = eligibility(ORDERS["1045"], "RS-3020-BLU", 1, TODAY)

    assert result["eligible"] is False
    assert "within 30 days" in result["reasons"][0]


def test_electronics_have_a_shorter_window():
    # 15 days after delivery: fine for shoes, too late for a watch.
    result = eligibility(ORDERS["1047"], "EL-0900-BLK", 1, TODAY)

    assert result["eligible"] is False
    assert "within 10 days" in result["reasons"][0]


def test_non_returnable_category_in_a_mixed_order():
    shoes = eligibility(ORDERS["1046"], "RS-3020-BLU", 2, TODAY)
    insoles = eligibility(ORDERS["1046"], "AC-0031-BLK", 1, TODAY)

    assert shoes["eligible"] is True
    assert shoes["refund_amount"] == 2 * 4499
    assert insoles["eligible"] is False
    assert "hygiene" in insoles["reasons"][0]


@pytest.mark.parametrize(
    "order_id, sku, expected",
    [
        ("1044", "AC-0031-BLK", "can be cancelled instead"),
        ("1042", "RS-4471-BLK", "hasn't been delivered yet"),
    ],
)
def test_undelivered_orders_say_what_to_do_instead(order_id, sku, expected):
    result = eligibility(ORDERS[order_id], sku, 1, TODAY)

    assert result["eligible"] is False
    assert expected in result["reasons"][0]


def test_quantity_over_what_was_bought():
    result = eligibility(ORDERS["1046"], "RS-3020-BLU", 3, TODAY)

    assert result["reasons"] == ["only 2 of 2 can be returned"]


def test_all_failing_rules_are_reported_together():
    # Late AND over quantity: both, so the customer hears it once.
    result = eligibility(ORDERS["1045"], "RS-3020-BLU", 5, TODAY)

    assert len(result["reasons"]) == 2


def test_missing_order_and_missing_line():
    assert eligibility(None, "X", 1, TODAY)["found"] is False
    wrong_sku = eligibility(ORDERS["1043"], "RS-4471-BLK", 1, TODAY)
    assert wrong_sku["found"] is False
    assert "JK-2210-GRN" in wrong_sku["why"]  # tells the model what IS there


# --- tools -----------------------------------------------------------------


def test_check_tool_uses_todays_date():
    result = check_return_eligibility.invoke({"order_id": "1043", "sku": "JK-2210-GRN"})

    assert result["eligible"] is True
    assert RETURNS == {}  # checking changes nothing


def test_create_return_records_it_and_reduces_what_is_left():
    args = {"order_id": "1046", "sku": "RS-3020-BLU", "quantity": 1, "reason": "too small"}
    first = create_return.invoke(args)

    assert first["created"] is True
    assert first["refund_amount"] == 4499
    assert RETURNS[first["return_id"]]["status"] == "awaiting_item"
    # One of two pairs is already going back.
    again = eligibility(ORDERS["1046"], "RS-3020-BLU", 2, TODAY)
    assert again["reasons"] == ["only 1 of 2 can be returned (1 already on an open return)"]


def test_create_return_rechecks_and_refuses_an_ineligible_item():
    # The model skipped check_return_eligibility. The rules still apply.
    result = create_return.invoke(
        {"order_id": "1046", "sku": "AC-0031-BLK", "quantity": 1, "reason": "didn't fit"}
    )

    assert result["created"] is False
    assert "hygiene" in result["reasons"][0]
    assert RETURNS == {}


def test_escalate_opens_a_ticket():
    result = escalate_to_human.invoke(
        {"summary": "wants a refund on 1045, outside window", "reason": "disputes policy"}
    )

    assert result["ticket_id"].startswith("ESC-")
    assert result["priority"] == "normal"
    assert "summary" not in result
    assert ESCALATIONS[0]["summary"] == "wants a refund on 1045, outside window"


def test_escalate_priority_is_an_enum_in_the_schema():
    # Literal["normal", "urgent"] becomes a JSON Schema enum, so Claude is
    # told the only two values it may send.
    schema = escalate_to_human.tool_call_schema.model_json_schema()

    assert schema["properties"]["priority"]["enum"] == ["normal", "urgent"]


# --- in the graph ----------------------------------------------------------


def return_then(answer: str) -> list[AIMessage]:
    args = {"order_id": "1043", "sku": "JK-2210-GRN", "quantity": 1, "reason": "wrong colour"}
    return [AIMessage("", tool_calls=[call("create_return", args)]), AIMessage(answer)]


def test_create_return_waits_for_approval():
    graph = build_explicit(scripted(*return_then("Done.")), checkpointer=InMemorySaver())
    graph.invoke(turn_input("return my jacket"), thread("t"))

    [pending] = pending_approval(graph, thread("t"))
    assert pending["name"] == "create_return"
    assert RETURNS == {}

    graph.invoke(decision(True), thread("t"))
    assert len(RETURNS) == 1


def test_rejected_return_creates_nothing():
    graph = build_explicit(scripted(*return_then("Not possible.")), checkpointer=InMemorySaver())
    graph.invoke(turn_input("return my jacket"), thread("t"))

    graph.invoke(decision(False, "item flagged as worn"), thread("t"))

    assert RETURNS == {}


def test_checks_and_escalations_do_not_wait_for_approval():
    model = scripted(
        AIMessage(
            "",
            tool_calls=[
                call("check_return_eligibility", {"order_id": "1045", "sku": "RS-3020-BLU"}, id="c1"),
                call("escalate_to_human", {"summary": "s", "reason": "r"}, id="c2"),
            ],
        ),
        AIMessage("Ticket opened."),
    )
    graph = build_explicit(model, checkpointer=InMemorySaver())
    graph.invoke(turn_input("I want a refund"), thread("t"))

    assert pending_approval(graph, thread("t")) == []
    assert len(ESCALATIONS) == 1
