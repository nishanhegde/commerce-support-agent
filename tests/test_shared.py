"""Day 53 tests: the shared case file, its conflict rules, and shared mode in
the supervisor graph. Scripted models, no API calls."""

from datetime import date
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from support_agent import returns
from support_agent.audit import AuditLog, format_row
from support_agent.case import check_writes, merge_case, update_case_tool
from support_agent.persistence import thread
from support_agent.supervisor import build_supervisor, render, turn_input
from test_memory import RecordingModel
from test_react import call, scripted
from test_supervisor import finish, route


@pytest.fixture(autouse=True)
def pinned_today(monkeypatch):
    monkeypatch.setattr(returns, "today", lambda: date(2026, 10, 27))


def write(by, notes=(), **fields):
    return {"by": by, "fields": fields, "notes": list(notes)}


def record(id="u1", notes=(), **fields):
    return AIMessage("", tool_calls=[call("update_case", {"fields": fields, "notes": list(notes)}, id=id)])


# --- the rules ---------------------------------------------------------------


def test_owner_write_is_accepted():
    case = merge_case(None, write("order_agent", order_id="1048", days_late=9))

    assert case["fields"]["days_late"] == {"value": 9, "by": "order_agent"}
    assert case["conflicts"] == []


def test_non_owner_write_is_rejected_and_recorded():
    case = merge_case(None, write("returns_agent", order_status="delivered"))

    assert "order_status" not in case["fields"]
    assert case["conflicts"] == [{"by": "returns_agent", "field": "order_status", "why": "owned by order_agent"}]


def test_owner_overwrite_keeps_history():
    case = merge_case(None, write("order_agent", order_status="in_transit"))
    case = merge_case(case, write("order_agent", order_status="delivered"))

    assert case["fields"]["order_status"]["value"] == "delivered"
    assert case["history"] == [{"field": "order_status", "old": "in_transit", "new": "delivered", "by": "order_agent"}]


def test_unknown_fields_are_rejected():
    _, rejected = check_writes("returns_agent", {"refund_amt": 7999})

    assert rejected["refund_amt"].startswith("unknown field")


def test_notes_are_open_to_everyone_and_append_only():
    case = merge_case(None, write("order_agent", notes=["customer is upset"]))
    case = merge_case(case, write("general_agent", notes=["prefers email"]))

    assert [n["note"] for n in case["notes"]] == ["customer is upset", "prefers email"]


def test_merge_never_mutates_the_old_case():
    before = merge_case(None, write("order_agent", order_id="1048"))
    snapshot = repr(before)
    merge_case(before, write("order_agent", order_id="1042", notes=["x"]))

    assert repr(before) == snapshot


def test_disjoint_writes_commute():
    a, b = write("order_agent", days_late=9), write("returns_agent", refund_amount=7999)

    one, two = merge_case(merge_case(None, a), b), merge_case(merge_case(None, b), a)
    assert one["fields"] == two["fields"]


def test_parallel_writers_need_the_reducer():
    # Two nodes in the same superstep both write `case`. With merge_case as
    # the reducer, LangGraph applies both; without one it would raise
    # InvalidUpdateError (Day 45's rule).
    class S(TypedDict, total=False):
        case: Annotated[dict, merge_case]

    g = StateGraph(S)
    g.add_node("orders", lambda s: {"case": write("order_agent", days_late=9)})
    g.add_node("refunds", lambda s: {"case": write("returns_agent", refund_amount=7999)})
    g.add_edge(START, "orders")
    g.add_edge(START, "refunds")
    g.add_edge("orders", END)
    g.add_edge("refunds", END)

    case = g.compile().invoke({})["case"]
    assert set(case["fields"]) == {"days_late", "refund_amount"}


def test_tool_tells_the_agent_what_was_rejected():
    tool = update_case_tool("returns_agent")
    result = tool.invoke({"fields": {"refund_amount": 7999, "order_status": "delivered"}})

    assert result == {"recorded": ["refund_amount"], "notes_added": 0, "rejected": {"order_status": "owned by order_agent"}}


# --- shared mode in the graph ----------------------------------------------


def shared_graph(sup, audit=None, **workers):
    return build_supervisor(sup, worker_models=workers, checkpointer=InMemorySaver(),
                            audit=audit or AuditLog(), shared=True)


def two_workers(sup, returns_model, audit=None):
    return shared_graph(
        sup,
        audit=audit,
        order_agent=scripted(
            AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1048"}, id="s1")]),
            record(order_id="1048", order_status="in_transit", days_late=9),
            AIMessage("1048 is 9 days late."),
        ),
        returns_agent=returns_model,
    )


def test_a_worker_reads_the_conversation_and_the_case_file():
    receiver = RecordingModel(messages=iter([AIMessage("Eligible, ₹7,999.")]))
    graph = two_workers(
        scripted(route("order_agent", "Status of the late order?", id="a"),
                 route("returns_agent", "Can it be refunded?", id="b"), finish("ok")),
        receiver,
    )
    graph.invoke(turn_input("My Trail Runner is a week late. I want my money back!"), thread("t"))

    brief = receiver.prompts[0][1].content
    assert brief.startswith("Supervisor's task: Can it be refunded?")
    assert "customer: My Trail Runner is a week late. I want my money back!" in brief
    assert '- days_late: 9 (from order_agent)' in brief  # what the order agent established


def test_the_supervisor_reads_the_case_file_too():
    sup = RecordingModel(messages=iter([route("order_agent", "Status?"), finish("ok")]))
    graph = shared_graph(sup, order_agent=scripted(record(order_id="1048"), AIMessage("done")))
    graph.invoke(turn_input("where is it"), thread("t"))

    assert '- order_id: "1048" (from order_agent)' in sup.prompts[1][0].content


def test_a_conflicting_write_is_kept_out_and_logged():
    audit = AuditLog()
    graph = two_workers(
        scripted(route("order_agent", "Status?", id="a"), route("returns_agent", "Refundable?", id="b"), finish("ok")),
        scripted(record(refund_eligible=True, refund_amount=7999, order_status="refund_pending"),
                 AIMessage("Eligible.")),
        audit=audit,
    )
    graph.invoke(turn_input("refund"), thread("t"))

    case = graph.get_state(thread("t")).values["case"]
    assert case["fields"]["order_status"] == {"value": "in_transit", "by": "order_agent"}  # unchanged
    assert case["fields"]["refund_amount"]["value"] == 7999
    [row] = audit.rows("t", "conflict")
    assert row["detail"]["field"] == "order_status"
    assert "CONFLICT rejected      returns_agent wrote order_status=\"refund_pending\"  (owned by order_agent)" in format_row(row)


def test_the_case_file_outlives_the_turn():
    graph = shared_graph(
        scripted(route("order_agent", "Status?"), finish("It's late."), finish("You're welcome.")),
        order_agent=scripted(record(order_id="1048"), AIMessage("done")),
    )
    graph.invoke(turn_input("where is it"), thread("t"))
    graph.invoke(turn_input("thanks"), thread("t"))

    state = graph.get_state(thread("t")).values
    assert state["reports"] == []  # per turn
    assert state["case"]["fields"]["order_id"]["value"] == "1048"  # per case


def test_isolated_mode_has_no_case_tool_and_no_conversation():
    receiver = RecordingModel(messages=iter([AIMessage("ok")]))
    graph = build_supervisor(scripted(route("order_agent", "Status?"), finish("ok")),
                             worker_models={"order_agent": receiver}, checkpointer=InMemorySaver())
    graph.invoke(turn_input("my order is late!"), thread("t"))

    system, brief = receiver.prompts[0]
    assert brief.content == "Status?"
    assert "update_case" not in system.content


def test_render_shows_case_writes():
    delta = {"case": write("order_agent", notes=["upset"], order_id="1048"), "reports": [{"report": "done"}]}

    assert render("order_agent", delta)[0] == "  order_agent wrote to case: order_id (+1 notes)"
