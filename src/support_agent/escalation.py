"""Day 54: escalation done right. Summarise, package, transfer.

Day 47's escalate_to_human was a stub: the model wrote a one-line summary
and got a ticket id back. A human agent picking that up would have to
re-read everything or, worse, ask the customer to repeat it. This module
builds what they actually need, in three steps:

    1. summarise   one model call turns the conversation into a structured
                   summary: the issue, what the customer wants, their mood,
                   what's been done, what to do next, what's still open.
    2. package     the handoff document: that summary, plus facts copied
                   from state and the audit log, not generated: customer
                   profile, case file, every action, approval and handoff,
                   and the full transcript.
    3. transfer    file it in the queue, write it where a human agent's
                   desktop would pick it up (data/handoffs/<ticket>.md and
                   .json), and take the bot out of the conversation.

The split in step 2 matters most. The summary is the model's reading of the
case and may be wrong; it's labelled as AI-generated and sits above the
facts it can be checked against. The facts are never paraphrased, so a
human can act on them: refund ₹7,999 approved by priya under
refund_over_threshold is a record, not the model's recollection.

"Take the bot out" is step 3's other half. Once a case is transferred, the
supervisor graph stops answering on that thread: new customer messages are
added to the ticket and acknowledged, with no model call. Two agents
answering one customer, one of them unaware of what the other promised, is
the failure that rule prevents.
"""

import itertools
import json
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from support_agent.react import message_text
from support_agent.returns import ESCALATIONS

HANDOFF_DIR = Path(__file__).resolve().parents[2] / "data" / "handoffs"

_ticket_seq = itertools.count(1)


class HandoffSummary(BaseModel):
    """Summarise this support conversation for the human agent taking it over."""

    issue: str = Field(description="The problem in one or two sentences, with order numbers.")
    customer_wants: str = Field(description="The outcome the customer is asking for, in their terms.")
    sentiment: Literal["calm", "frustrated", "angry"] = Field(description="The customer's mood right now.")
    already_done: list[str] = Field(description="What the agents checked or did, each with its result.")
    suggested_next_step: str = Field(description="The one thing the human should do first.")
    open_questions: list[str] = Field(default_factory=list, description="Anything still unknown or undecided.")


SUMMARY_SYSTEM = """You are writing the handover note for a human support
agent who is taking over this conversation from an AI assistant. They will
read your note first, so they don't make the customer repeat themselves.

Stick to what the conversation and the record below show. Don't guess
order details, amounts or outcomes; if something is unclear, put it under
open_questions. Call HandoffSummary once."""


def summarise(model: BaseChatModel, transcript: list[str], facts: str) -> HandoffSummary:
    """Step 1. Forced tool call, so the summary arrives as validated fields."""
    response = model.bind_tools([HandoffSummary], tool_choice="HandoffSummary").invoke(
        [SystemMessage(SUMMARY_SYSTEM + "\n\n" + facts), ("user", "Conversation:\n" + "\n".join(transcript))]
    )
    if response.tool_calls:
        return HandoffSummary(**response.tool_calls[0]["args"])
    # Shouldn't happen with the tool forced. If it does, hand over anyway:
    # a human with the facts and no summary beats no handover.
    return HandoffSummary(
        issue=message_text(response) or "(summary unavailable)", customer_wants="see transcript",
        sentiment="frustrated", already_done=[], suggested_next_step="read the transcript",
    )


def transcript_lines(messages: list) -> list[str]:
    return [f"{'customer' if m.type == 'human' else 'agent'}: {message_text(m)}" for m in messages]


def actions_from_audit(rows: list[dict]) -> list[str]:
    """Every action, approval, handoff and conflict on the thread, verbatim
    from the audit log. Lookups are left out: they changed nothing."""
    # A rejected call still gets an error ToolMessage (every tool_use needs
    # a result), but the tool never ran. Saying "ran (error)" would read as
    # a system failure, so those results are left to the rejection line.
    rejected = {r["detail"]["tool_call_id"] for r in rows if r["kind"] == "approval" and r["outcome"] == "rejected"}
    actions = []
    for r in rows:
        d, at = r["detail"], r["at"][11:19]
        if r["kind"] == "approval":
            who = f"policy rule {d['rule']}" if d["by"] == "policy" else d["by"]
            reason = f"; reason: {d['reason']}" if d.get("reason") else ""
            actions.append(f"{at} {d['tool']}({json.dumps(d['args'], ensure_ascii=False)}) "
                           f"{r['outcome']} by {who} ({d['why']}{reason})")
        elif r["kind"] == "handoff":
            actions.append(f"{at} {d['from']} handed off to {d['to']} ({r['outcome']}): {d['reason']}")
        elif r["kind"] == "conflict":
            actions.append(f"{at} {r['node']} tried to set {d['field']}: rejected, {d['why']}")
        elif r["kind"] == "transition":
            for t in d.get("tool_results", []):
                if t["name"] in ("create_return", "cancel_order") and t["id"] not in rejected:
                    actions.append(f"{at} {t['name']} ran ({t['status']})")
    return actions


def package(
    ticket_id: str,
    reason: str,
    priority: str,
    customer: dict | None,
    summary: HandoffSummary,
    case: dict | None,
    actions: list[str],
    transcript: list[str],
) -> dict:
    """Step 2. Everything as plain data, which the .md and .json both come from."""
    return {
        "ticket_id": ticket_id,
        "queue": "support-tier2",
        "status": "open",
        "priority": priority,
        "reason": reason,
        "customer": {k: v for k, v in (customer or {}).items() if k != "notes"},
        "customer_notes": (customer or {}).get("notes", []),
        "summary": summary.model_dump(),
        "case_file": {k: f["value"] for k, f in (case or {}).get("fields", {}).items()},
        "actions": actions,
        "transcript": transcript,
        "after_transfer": [],
    }


def render_markdown(doc: dict) -> str:
    """The handoff document as a human agent reads it."""
    c, s = doc["customer"], doc["summary"]
    lines = [f"# {doc['ticket_id']}: {doc['reason']}", "",
             f"**Priority:** {doc['priority']} · **Queue:** {doc['queue']} · **Status:** {doc['status']}", ""]
    lines += ["## Customer", ""]
    if c.get("name"):
        lines.append(f"{c['name']} ({c['customer_id']}), {c['tier']} tier since {c['member_since']}. "
                     f"Prefers contact by {c['preferred_contact']}.")
        orders = ", ".join(f"{o['order_id']} ({o['status']})" for o in c.get("past_orders", []))
        lines.append(f"Orders: {orders or 'none'}")
    else:
        lines.append("Not signed in.")
    lines += [f"- Told us before: {n}" for n in doc["customer_notes"]]
    lines += ["", "## Summary (AI-generated: check it against the record below)", "",
              f"- **Issue:** {s['issue']}",
              f"- **Customer wants:** {s['customer_wants']}",
              f"- **Mood:** {s['sentiment']}",
              f"- **Suggested next step:** {s['suggested_next_step']}"]
    lines += ["- **Already done:**", *[f"  - {x}" for x in s["already_done"] or ["nothing"]]]
    if s["open_questions"]:
        lines += ["- **Open questions:**", *[f"  - {q}" for q in s["open_questions"]]]
    lines += ["", "## Record (copied from the system, not generated)", ""]
    if doc["case_file"]:
        lines += ["Case file:", *[f"- {k}: {json.dumps(v, ensure_ascii=False)}" for k, v in doc["case_file"].items()], ""]
    lines += ["Actions:", *[f"- {a}" for a in doc["actions"] or ["none"]], ""]
    lines += ["## Transcript", "", *[f"> {t}" for t in doc["transcript"]]]
    if doc["after_transfer"]:
        lines += ["", "## Messages since transfer", "", *[f"> customer: {t}" for t in doc["after_transfer"]]]
    return "\n".join(lines) + "\n"


def transfer(doc: dict, handoff_dir: Path | None) -> None:
    """Step 3. Queue it and write it out. `handoff_dir=None` queues only
    (tests that don't care about files)."""
    ESCALATIONS.append(doc)
    _write(doc, handoff_dir)


def add_after_transfer(ticket_id: str, text: str, handoff_dir: Path | None) -> dict | None:
    """A customer message that arrived after the transfer goes onto the
    ticket, not to a model."""
    doc = next((d for d in ESCALATIONS if d["ticket_id"] == ticket_id), None)
    if doc is not None:
        doc["after_transfer"].append(text)
        _write(doc, handoff_dir)
    return doc


def _write(doc: dict, handoff_dir: Path | None) -> None:
    if handoff_dir is None:
        return
    handoff_dir.mkdir(parents=True, exist_ok=True)
    (handoff_dir / f"{doc['ticket_id']}.md").write_text(render_markdown(doc))
    (handoff_dir / f"{doc['ticket_id']}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))


def next_ticket_id() -> str:
    return f"HO-{next(_ticket_seq):04d}"


def transferred_reply(ticket_id: str) -> AIMessage:
    # F3's rule: say a person has it; promise nothing about how or when.
    return AIMessage(
        f"I've passed your case to a member of our support team (reference {ticket_id}). "
        "They'll see everything we've discussed and everything already done, so you won't need to repeat it."
    )


def holding_reply(ticket_id: str) -> AIMessage:
    return AIMessage(f"Your case is with our support team (reference {ticket_id}). I've added this message to it.")
