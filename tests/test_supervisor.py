"""Day 51 tests: the supervisor routes, workers report, the supervisor
decides when it's done. Each model is scripted, so no API calls."""

from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from support_agent import returns
from support_agent.audit import AuditLog
from support_agent.hitl import decision, pending_approval
from support_agent.memory import CustomerContext, seed_profiles
from support_agent.persistence import thread
from support_agent.returns import RETURNS
from support_agent.supervisor import WORKERS, Route, _parse, build_supervisor, turn_input
from test_memory import RecordingModel
from test_react import call, scripted

LATE_ORDER = {"order_id": "1048", "sku": "RS-4471-BLK", "quantity": 1, "reason": "9 days late"}


@pytest.fixture(autouse=True)
def pinned_today(monkeypatch):
    monkeypatch.setattr(returns, "today", lambda: date(2026, 10, 27))


def route(next, task="", reply="", id="r"):
    return AIMessage("", tool_calls=[call("Route", {"next": next, "task": task, "reply": reply}, id=id)])


def finish(reply):
    return route("FINISH", reply=reply, id="f")


def steps(graph, graph_input, config, context=None):
    return [n for u in graph.stream(graph_input, config, stream_mode="updates", context=context) for n in u]


def supervisor_graph(sup, audit=None, store=None, **workers):
    return build_supervisor(
        sup, worker_models=workers, checkpointer=InMemorySaver(), store=store, audit=audit or AuditLog()
    )


# --- routing ---------------------------------------------------------------


def test_small_talk_finishes_without_a_worker():
    graph = supervisor_graph(scripted(finish("Hello! How can I help?")))

    assert steps(graph, turn_input("hi"), thread("t")) == ["supervisor"]
    assert graph.get_state(thread("t")).values["messages"][-1].content == "Hello! How can I help?"


def test_delegate_report_finish():
    graph = supervisor_graph(
        scripted(route("order_agent", "Status of order 1042?"), finish("1042 arrives tomorrow.")),
        order_agent=scripted(
            AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1042"})]),
            AIMessage("1042: in transit with BlueDart, ETA 2026-10-28."),
        ),
    )

    assert steps(graph, turn_input("where's 1042?"), thread("t")) == ["supervisor", "order_agent", "supervisor"]
    state = graph.get_state(thread("t")).values
    assert state["reports"] == [
        {"worker": "order_agent", "task": "Status of order 1042?",
         "report": "1042: in transit with BlueDart, ETA 2026-10-28."}
    ]


def test_the_conversation_holds_only_what_the_customer_sees():
    graph = supervisor_graph(
        scripted(route("order_agent", "Status of 1042?"), finish("Arrives tomorrow.")),
        order_agent=scripted(
            AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1042"})]),
            AIMessage("In transit."),
        ),
    )
    graph.invoke(turn_input("where's 1042?"), thread("t"))

    messages = graph.get_state(thread("t")).values["messages"]
    assert [m.type for m in messages] == ["human", "ai"]
    assert not any(isinstance(m, ToolMessage) for m in messages)


def test_a_worker_sees_only_its_brief():
    worker = RecordingModel(messages=iter([AIMessage("Nothing to do.")]))
    graph = supervisor_graph(
        scripted(route("general_agent", "Is the Road Runner 2 in stock?"), finish("Yes.")),
        general_agent=worker,
    )
    graph.invoke(turn_input("I'm so annoyed. Do you have Road Runner 2?"), thread("t"))

    system, *rest = worker.prompts[0]
    assert "general_agent" in system.content
    assert [m.content for m in rest] == ["Is the Road Runner 2 in stock?"]  # not the customer's words


def test_the_supervisor_sees_each_report_before_deciding_again():
    sup = RecordingModel(messages=iter([route("order_agent", "Status of 1042?"), finish("Tomorrow.")]))
    graph = supervisor_graph(sup, order_agent=scripted(AIMessage("ETA 2026-10-28.")))
    graph.invoke(turn_input("where's 1042?"), thread("t"))

    first, second = (p[0].content for p in sup.prompts)
    assert "Work done this turn" not in first
    assert "order_agent was asked: Status of 1042?\n  reported: ETA 2026-10-28." in second


def test_aggregates_two_workers_into_one_reply():
    graph = supervisor_graph(
        scripted(
            route("order_agent", "Status of 1048?", id="r1"),
            route("returns_agent", "Can 1048 be refunded? Don't open anything yet.", id="r2"),
            finish("1048 is 9 days late; you can have a ₹7,999 refund. Shall I?"),
        ),
        order_agent=scripted(AIMessage("1048: in transit, ETA 2026-10-18.")),
        returns_agent=scripted(AIMessage("Eligible: delayed shipment, ₹7,999.")),
    )

    assert steps(graph, turn_input("my order is late, refund me"), thread("t")) == [
        "supervisor", "order_agent", "supervisor", "returns_agent", "supervisor",
    ]
    assert [r["worker"] for r in graph.get_state(thread("t")).values["reports"]] == ["order_agent", "returns_agent"]


# --- deciding when it's done -----------------------------------------------


def test_delegation_limit_forces_a_finish():
    graph = build_supervisor(
        scripted(*(route("order_agent", "again", id=f"r{i}") for i in range(3))),
        worker_models={"order_agent": scripted(AIMessage("a"), AIMessage("b"))},
        checkpointer=InMemorySaver(),
        max_delegations=2,
    )
    graph.invoke(turn_input("x"), thread("t"))

    state = graph.get_state(thread("t")).values
    assert state["delegations"] == 2
    assert "couldn't finish" in state["messages"][-1].content


def test_each_turn_starts_with_no_reports():
    graph = supervisor_graph(
        scripted(route("order_agent", "1042?"), finish("Tomorrow."), finish("You're welcome.")),
        order_agent=scripted(AIMessage("ETA tomorrow.")),
    )
    graph.invoke(turn_input("where's 1042?"), thread("t"))
    graph.invoke(turn_input("thanks"), thread("t"))

    state = graph.get_state(thread("t")).values
    assert state["reports"] == [] and state["delegations"] == 0


def test_a_text_reply_without_route_is_treated_as_finish():
    assert _parse(AIMessage("Just text.")) == Route(next="FINISH", reply="Just text.")


# --- guard rails inside workers ----------------------------------------------


def refund_graph(audit, approve_reply):
    return supervisor_graph(
        scripted(
            route("returns_agent", "Customer confirmed: refund order 1048, 1 x RS-4471-BLK."),
            finish(approve_reply),
        ),
        audit=audit,
        returns_agent=scripted(
            AIMessage("", tool_calls=[call("create_return", LATE_ORDER, id="c1")]),
            AIMessage("Done: RMA-1048-1, ₹7,999, carrier recall."),
        ),
    )


def test_an_approval_inside_a_worker_pauses_the_whole_graph():
    audit = AuditLog()
    graph = refund_graph(audit, "Refund approved: RMA-1048-1.")
    graph.invoke(turn_input("yes, refund it"), thread("t"))

    [pending] = pending_approval(graph, thread("t"))
    assert pending["name"] == "create_return"
    assert pending["why"] == "refund ₹7,999 is over the ₹5,000 limit"
    assert RETURNS == {}

    # The worker's script has exactly two replies. If resuming re-ran its
    # first model call, the script would run out and this would raise.
    graph.invoke(decision(True, reviewer="priya"), thread("t"))

    assert len(RETURNS) == 1
    assert graph.get_state(thread("t")).values["messages"][-1].content == "Refund approved: RMA-1048-1."
    [row] = audit.rows("t", "approval")
    assert (row["outcome"], row["detail"]["by"]) == ("approved", "priya")


def test_a_rejection_inside_a_worker_reaches_the_supervisor_as_a_report():
    graph = supervisor_graph(
        scripted(route("returns_agent", "Refund 1048, confirmed."), finish("The refund wasn't approved.")),
        returns_agent=scripted(
            AIMessage("", tool_calls=[call("create_return", LATE_ORDER, id="c1")]),
            AIMessage("The reviewer rejected the refund: carrier says out for delivery."),
        ),
    )
    graph.invoke(turn_input("refund it"), thread("t"))
    graph.invoke(decision(False, "carrier says out for delivery", reviewer="priya"), thread("t"))

    assert RETURNS == {}
    [report] = graph.get_state(thread("t")).values["reports"]
    assert "rejected" in report["report"]


def test_audit_rows_name_the_worker():
    audit = AuditLog()
    graph = refund_graph(audit, "Done.")
    graph.invoke(turn_input("yes, refund it"), thread("t"))
    graph.invoke(decision(True, reviewer="priya"), thread("t"))

    # The pause travels up: the worker's `approve`, then the parent's node
    # that called the worker. On resume the worker picks up at `approve`;
    # its first `agent` step (the model call) is not run again.
    assert [(r["node"], r["outcome"]) for r in audit.rows("t", "transition")] == [
        ("supervisor", "completed"),
        ("returns_agent/agent", "completed"),
        ("returns_agent/approve", "paused"),
        ("returns_agent", "paused"),
        ("returns_agent/approve", "completed"),
        ("returns_agent/tools", "completed"),
        ("returns_agent/agent", "completed"),
        ("returns_agent", "completed"),
        ("supervisor", "completed"),
    ]


def test_each_write_tool_belongs_to_exactly_one_worker():
    writes = {"cancel_order", "create_return", "save_customer_note"}
    owners = {w: [n for n, (tools, _) in WORKERS.items() if w in {t.name for t in tools}] for w in writes}

    assert all(len(o) == 1 for o in owners.values()), owners
    # Day 54: escalation is the supervisor's decision, so no worker has it.
    assert not any("escalate_to_human" in {t.name for t in tools} for tools, _ in WORKERS.values())


# --- memory ------------------------------------------------------------------


def test_customer_is_loaded_once_and_passed_to_workers():
    store = InMemoryStore()
    seed_profiles(store)
    worker = RecordingModel(messages=iter([AIMessage("1048 is late.")]))
    graph = supervisor_graph(
        scripted(route("order_agent", "Status of 1048?"), finish("It's late.")), store=store, order_agent=worker
    )

    assert steps(graph, turn_input("where's my order?"), thread("t"), CustomerContext("C-100"))[0] == "load_customer"
    assert "Name: Asha Rao" in worker.prompts[0][0].content
