"""Day 53: shared state. One case file every agent reads, and writes by rules.

Day 51's workers were isolated: a brief in, a report out, nothing shared.
In shared mode (build_supervisor(shared=True)) every agent reads the same
two things:

    the conversation   what the customer said and was told. Read by all;
                       written only by the supervisor, the one voice the
                       customer hears.
    the case file      facts established so far: which order, how late,
                       eligible or not, return id. Read by all; written by
                       workers through the update_case tool.

The conflict rules, for the case file:

    1. One writer per field. Each field has an owner, the agent whose tools
       are its source of truth: order facts belong to order_agent, refund
       facts to returns_agent. Another agent writing it is a conflict: the
       write is rejected, the agent is told, and it's logged. It's the
       system-of-record rule: OMS owns order status, nobody else sets it.
    2. The owner may overwrite, and the old value goes into `history`.
       Last write wins, but nothing is lost silently.
    3. Notes are append-only and open to all. Two appends never conflict,
       whatever order they arrive in.
    4. Unknown fields are rejected. A case file anyone can add any key to
       becomes a junk drawer, and a misspelt field ("refund_amt") would
       silently miss its owner check.

Where the rules are enforced: `merge_case` is the case field's reducer, the
only way a write gets into state. The update_case tool checks the same rules
first, so the agent hears "rejected" while it can still adjust, but the
reducer is the guarantee. That also makes the rules hold for parallel
writers: LangGraph applies every write in a step through the reducer, and
disjoint ownership means the order they're applied in can't change the
result. (Without a reducer, two parallel writes to one key is an
InvalidUpdateError, as on Day 45.)
"""

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

FIELD_OWNERS = {
    "order_id": "order_agent",
    "order_status": "order_agent",
    "days_late": "order_agent",
    "refund_eligible": "returns_agent",
    "refund_amount": "returns_agent",
    "return_id": "returns_agent",
    "product_answer": "general_agent",
}


def check_writes(writer: str, fields: dict) -> tuple[dict, dict]:
    """Split a write into (accepted, rejected). rejected maps field -> why."""
    accepted, rejected = {}, {}
    for key, value in fields.items():
        owner = FIELD_OWNERS.get(key)
        if owner is None:
            rejected[key] = f"unknown field; allowed: {sorted(FIELD_OWNERS)}"
        elif owner != writer:
            rejected[key] = f"owned by {owner}"
        else:
            accepted[key] = value
    return accepted, rejected


def merge_case(case: dict | None, write: dict) -> dict:
    """The reducer. `case` is the case file so far; `write` is one agent's
    {"by": ..., "fields": {...}, "notes": [...]}. Returns a new case file;
    the old one is never mutated, which is how reducers must behave."""
    case = case or {"fields": {}, "history": [], "notes": [], "conflicts": []}
    by = write["by"]
    accepted, rejected = check_writes(by, write.get("fields", {}))
    fields = dict(case["fields"])
    history = list(case["history"])
    for key, value in accepted.items():
        if key in fields and fields[key]["value"] != value:
            history.append({"field": key, "old": fields[key]["value"], "new": value, "by": by})
        fields[key] = {"value": value, "by": by}
    return {
        "fields": fields,
        "history": history,
        "notes": [*case["notes"], *({"by": by, "note": n} for n in write.get("notes", []))],
        "conflicts": [*case["conflicts"], *({"by": by, "field": k, "why": w} for k, w in rejected.items())],
    }


def case_block(case: dict | None) -> str:
    """The case file as prompt text, for every agent."""
    if not case or not (case["fields"] or case["notes"]):
        return "## Case file\n(empty: nothing established yet)"
    lines = ["## Case file (shared; facts other agents established)"]
    lines += [f"- {k}: {json.dumps(f['value'], ensure_ascii=False)} (from {f['by']})" for k, f in case["fields"].items()]
    lines += [f"- note from {n['by']}: {n['note']}" for n in case["notes"]]
    return "\n".join(lines)


class CaseUpdate(BaseModel):
    fields: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Facts to record, by field name. Only fields you own are accepted.",
    )
    notes: list[str] = Field(default_factory=list, description="Free-text notes for the team; anyone may add.")


def update_case_tool(writer: str) -> StructuredTool:
    """update_case for one agent. Bound to its writer, so it can't claim to
    be someone else: the owner check uses `writer`, not anything the model
    sends."""
    owned = sorted(k for k, o in FIELD_OWNERS.items() if o == writer)

    def update_case(fields: dict | None = None, notes: list[str] | None = None) -> dict:
        accepted, rejected = check_writes(writer, fields or {})
        return {"recorded": sorted(accepted), "notes_added": len(notes or []), "rejected": rejected}

    return StructuredTool.from_function(
        update_case,
        name="update_case",
        description=(
            "Record facts in the shared case file, which every agent and the supervisor read. "
            f"You own these fields: {', '.join(owned) or 'none'}; record them as soon as a tool "
            "establishes them. Use notes for anything else the team should know."
        ),
        args_schema=CaseUpdate,
    )


def writes_in(messages: list) -> list[dict]:
    """The update_case calls a worker made, in order, as their raw args."""
    return [
        c["args"]
        for m in messages
        for c in getattr(m, "tool_calls", None) or []
        if c["name"] == "update_case"
    ]
