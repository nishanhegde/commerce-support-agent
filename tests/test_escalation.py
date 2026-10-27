"""Day 54 tests: summarise, package, transfer, and the bot stepping out.
Scripted models, no API calls."""

import json
from datetime import date

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from support_agent import escalation, returns
from support_agent.audit import AuditLog, format_row
from support_agent.escalation import HandoffSummary, summarise
from support_agent.hitl import decision
from support_agent.memory import CustomerContext, seed_profiles
from support_agent.persistence import thread
from support_agent.returns import ESCALATIONS, RETURNS
from support_agent.supervisor import build_supervisor, render, turn_input
from test_memory import RecordingModel
from test_react import call, scripted
from test_supervisor import finish, route

ASHA = CustomerContext("C-100")
LATE_ORDER = {"order_id": "1048", "sku": "RS-4471-BLK", "quantity": 1, "reason": "9 days late"}
SUMMARY = {
    "issue": "Order 1048 (Trail Runner 4) is 9 days late; refund of ₹7,999 was rejected by a reviewer.",
    "customer_wants": "A refund now.",
    "sentiment": "angry",
    "already_done": ["Checked 1048: eligible for a delayed-shipment refund", "Refund rejected by priya"],
    "suggested_next_step": "Check the carrier's latest scan before deciding on the refund.",
    "open_questions": ["Is the parcel really out for delivery?"],
}


@pytest.fixture(autouse=True)
def pinned(monkeypatch):
    monkeypatch.setattr(returns, "today", lambda: date(2026, 10, 27))
    monkeypatch.setattr(escalation, "_ticket_seq", iter(range(1, 100)))


def summary_model(**overrides):
    return scripted(AIMessage("", tool_calls=[call("HandoffSummary", {**SUMMARY, **overrides}, id="sum")]))


def escalate(reason="customer asked for a manager", priority="normal"):
    return AIMessage("", tool_calls=[call("Route", {"next": "ESCALATE", "task": reason, "priority": priority}, id="esc")])


def graph_for(sup, tmp_path, audit=None, summary=None, **workers):
    store = InMemoryStore()
    seed_profiles(store)
    return build_supervisor(
        sup, worker_models=workers, checkpointer=InMemorySaver(), store=store, audit=audit or AuditLog(),
        summary_model=summary or summary_model(), handoff_dir=tmp_path,
    )


def ask(graph, text, ctx=ASHA):
    return graph.invoke(turn_input(text), thread("t"), context=ctx)


# --- the transfer ------------------------------------------------------------


def test_escalate_transfers_and_tells_the_customer(tmp_path):
    graph = graph_for(scripted(escalate()), tmp_path)

    out = ask(graph, "I want to speak to a manager")

    reply = out["messages"][-1].content
    assert "HO-0001" in reply and "won't need to repeat" in reply
    assert "email" not in reply and "hour" not in reply  # no promised channel or time (F3)
    assert out["escalated"] == {"ticket_id": "HO-0001"}
    assert ESCALATIONS[-1]["ticket_id"] == "HO-0001"


def test_the_handoff_document_is_written_for_the_human(tmp_path):
    graph = graph_for(scripted(escalate(priority="urgent")), tmp_path)
    ask(graph, "I want to speak to a manager")

    doc = json.loads((tmp_path / "HO-0001.json").read_text())
    md = (tmp_path / "HO-0001.md").read_text()
    assert doc["priority"] == "urgent" and doc["reason"] == "customer asked for a manager"
    assert doc["customer"]["name"] == "Asha Rao"
    assert doc["transcript"] == ["customer: I want to speak to a manager"]
    assert "## Summary (AI-generated: check it against the record below)" in md
    assert "## Record (copied from the system, not generated)" in md
    assert "- **Mood:** angry" in md


def test_the_full_path_rejection_then_manager(tmp_path):
    audit = AuditLog()
    graph = graph_for(
        scripted(
            route("returns_agent", "Customer confirmed: refund 1048, 1 x RS-4471-BLK.", id="r1"),
            finish("I'm sorry, the refund wasn't approved."),
            escalate(),
        ),
        tmp_path,
        audit=audit,
        returns_agent=scripted(
            AIMessage("", tool_calls=[call("create_return", LATE_ORDER, id="c1")]),
            AIMessage("Reviewer rejected the refund."),
        ),
    )
    ask(graph, "refund my late order now")
    graph.invoke(decision(False, "carrier says out for delivery", reviewer="priya"), thread("t"), context=ASHA)
    ask(graph, "This is unacceptable. Get me a manager.")

    doc = ESCALATIONS[-1]
    # The rejection, verbatim from the audit log: who, under which rule, why.
    assert any("create_return" in a and "rejected by priya" in a and "carrier says out for delivery" in a
               for a in doc["actions"])
    # The rejected call's error result isn't reported as a run that failed.
    assert not any("ran (error)" in a for a in doc["actions"])
    assert doc["transcript"] == [
        "customer: refund my late order now",
        "agent: I'm sorry, the refund wasn't approved.",
        "customer: This is unacceptable. Get me a manager.",
    ]
    assert RETURNS == {}


def test_the_summary_is_written_from_the_facts(tmp_path):
    summariser = RecordingModel(messages=iter([AIMessage("", tool_calls=[call("HandoffSummary", SUMMARY, id="s")])]))
    graph = graph_for(scripted(escalate()), tmp_path, summary=summariser)
    ask(graph, "get me a person")

    system = summariser.prompts[0][0].content
    assert "Name: Asha Rao" in system
    assert "## Actions on record" in system


# --- the bot steps out -------------------------------------------------------


def test_after_transfer_messages_go_to_the_ticket_not_a_model(tmp_path):
    # The supervisor's script holds exactly one reply. A second model call
    # after the transfer would run it dry and raise.
    graph = graph_for(scripted(escalate()), tmp_path)
    ask(graph, "get me a person")

    out = ask(graph, "hello? anyone?")

    assert out["messages"][-1].content == (
        "Your case is with our support team (reference HO-0001). I've added this message to it."
    )
    assert ESCALATIONS[-1]["after_transfer"] == ["hello? anyone?"]
    assert "> customer: hello? anyone?" in (tmp_path / "HO-0001.md").read_text()


def test_steps_after_transfer(tmp_path):
    graph = graph_for(scripted(escalate()), tmp_path)
    ask(graph, "get me a person")

    updates = [n for u in graph.stream(turn_input("still there?"), thread("t"), stream_mode="updates", context=ASHA)
               for n in u]
    assert updates == ["with_human"]


# --- the trail ---------------------------------------------------------------


def test_escalation_is_audited(tmp_path):
    audit = AuditLog()
    graph = graph_for(scripted(escalate(priority="urgent")), tmp_path, audit=audit)
    ask(graph, "get me a person")
    ask(graph, "hello?")

    rows = audit.rows("t", "escalation")
    assert [r["outcome"] for r in rows] == ["transferred", "message_added"]
    assert "ESCALATE transferred   HO-0001  (urgent, customer angry): customer asked for a manager" in format_row(rows[0])


def test_shared_mode_puts_the_case_file_in_the_document(tmp_path):
    store = InMemoryStore()
    seed_profiles(store)
    graph = build_supervisor(
        scripted(route("order_agent", "Status of 1048?", id="r1"), escalate()),
        worker_models={"order_agent": scripted(
            AIMessage("", tool_calls=[call("update_case", {"fields": {"order_id": "1048", "days_late": 9}}, id="u1")]),
            AIMessage("1048 is 9 days late."),
        )},
        checkpointer=InMemorySaver(), store=store, audit=AuditLog(), shared=True,
        summary_model=summary_model(), handoff_dir=tmp_path,
    )
    ask(graph, "where's my order? get me a manager")

    assert ESCALATIONS[-1]["case_file"] == {"order_id": "1048", "days_late": 9}


# --- pieces ------------------------------------------------------------------


def test_summary_falls_back_when_the_model_skips_the_tool():
    s = summarise(scripted(AIMessage("Customer wants a refund.")), ["customer: hi"], "")

    assert isinstance(s, HandoffSummary) and s.issue == "Customer wants a refund."


def test_render_escalate_lines():
    assert render("supervisor", {"next": "ESCALATE", "priority": "normal", "task": "asked for a person"}) == [
        "  supervisor -> ESCALATE (normal): asked for a person"
    ]
