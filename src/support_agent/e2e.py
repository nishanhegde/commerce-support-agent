"""Day 50: the mid-project integration test, as a scripted scenario.

An angry customer, a delayed order, a refund demand. The customer's lines
are fixed; everything else (which tools, in what order, what to say) is up
to the agent. A reviewer stands in for the human at the approval prompt.

Afterwards, the run is judged in stages, each from evidence in the run:

    triage        the customer was loaded and the right order found
    order agent   the agent said plainly that the order is late
    eligibility   check_return_eligibility ran on that order, and said yes
    approval      the graph paused for a human, for the refund-threshold rule
    resolution    approved: a return exists, and the customer was told its id
                  rejected: no return, no escalation the customer didn't ask
                  for, and the customer was told it wasn't approved

There's no triage node or order specialist yet: one ReAct agent plays every
part. The stages are checked as behaviour in the trace, so the same checks
still apply when Days 51-52 split it into agents.

Shared by scripts/e2e.py (live Claude) and tests/test_e2e.py (a scripted
model replaying the path, so the checks themselves are tested offline).
"""

import re
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from support_agent.audit import AuditLog
from support_agent.hitl import decision, pending_approval
from support_agent.memory import CustomerContext, seed_profiles
from support_agent.persistence import thread, turn_input
from support_agent.react import build_explicit, message_text
from support_agent.returns import RETURNS
from support_agent.tracing import RunTrace

CUSTOMER = CustomerContext("C-100")
ORDER_ID = "1048"

# No order number on purpose: finding it is triage's job.
TURNS = [
    "This is ridiculous. My Trail Runner 4 was supposed to arrive over a week ago "
    "and it STILL isn't here. I don't want it anymore, I want my money back.",
    "Yes. Refund it.",
]


@dataclass
class Run:
    """Everything a run leaves behind, for the judge and for printing."""

    messages: list = field(default_factory=list)
    replies: list[str] = field(default_factory=list)  # the agent's answer per turn
    approvals: list[dict] = field(default_factory=list)  # what the reviewer was shown
    audit_rows: list[dict] = field(default_factory=list)


def run_scenario(
    model: BaseChatModel | None = None, approve: bool = True, audit: AuditLog | None = None, trace: RunTrace | None = None
) -> Run:
    """Play TURNS against a fresh graph. Fresh means in-memory checkpoints,
    Store and audit log, so a run never sees an earlier run's state."""
    store = InMemoryStore()
    seed_profiles(store)
    audit = audit or AuditLog()
    graph = build_explicit(model, checkpointer=InMemorySaver(), store=store, audit=audit)
    config = thread("e2e", *([trace] if trace else []))
    run = Run()

    for text in TURNS:
        graph_input = turn_input(text)
        while True:
            graph.invoke(graph_input, config, context=CUSTOMER)
            pending = pending_approval(graph, config)
            if not pending:
                break
            run.approvals.extend(pending)
            graph_input = decision(approve, "" if approve else "declined in e2e", reviewer="e2e-reviewer")
        run.replies.append(message_text(graph.get_state(config).values["messages"][-1]))

    run.messages = graph.get_state(config).values["messages"]
    run.audit_rows = audit.rows("e2e")
    return run


# --- the judge -------------------------------------------------------------


def tool_calls(run: Run, name: str) -> list[dict]:
    return [c for m in run.messages if isinstance(m, AIMessage) for c in m.tool_calls if c["name"] == name]


def tool_result(run: Run, call_id: str) -> str:
    return next((m.content for m in run.messages if isinstance(m, ToolMessage) and m.tool_call_id == call_id), "")


LATE_WORDS = re.compile(r"\b(late|delay(ed)?|overdue|past (its|the) (estimate|expected))\b", re.IGNORECASE)
DECLINED_WORDS = re.compile(
    r"\b(not (been )?approved|wasn't approved|declined|rejected|turned down|can't (issue|process))\b", re.IGNORECASE
)


def judge(run: Run, approved: bool = True) -> list[tuple[str, bool, str]]:
    """(stage, passed, evidence) for each stage, in order. `approved` is what
    the reviewer decided, so the judge knows which resolution is right."""
    loaded = any(r["node"] == "load_customer" and r["outcome"] == "completed" for r in run.audit_rows)
    looked_up = [c for c in tool_calls(run, "get_order_status") if c["args"].get("order_id") == ORDER_ID]
    checks = [c for c in tool_calls(run, "check_return_eligibility") if c["args"].get("order_id") == ORDER_ID]
    eligible = [c for c in checks if '"eligible": true' in tool_result(run, c["id"])]
    gated = [a for a in run.approvals if a["rule"] == "refund_over_threshold"]
    created = [r for r in RETURNS.values() if r["order_id"] == ORDER_ID]
    if approved:
        resolved = bool(created) and created[0]["return_id"] in run.replies[-1]
        resolution = f"returns for {ORDER_ID}={[r['return_id'] for r in created]}; id in last reply={resolved}"
    else:
        # TURNS never ask for a person, so any escalation is the agent's own.
        escalations = tool_calls(run, "escalate_to_human")
        said_no = bool(DECLINED_WORDS.search(run.replies[-1]))
        resolved = not created and not escalations and said_no
        resolution = f"returns={len(created)}; escalations={len(escalations)}; told it was declined={said_no}"

    return [
        ("triage", loaded and bool(looked_up),
         f"customer loaded={loaded}; get_order_status({ORDER_ID}) calls={len(looked_up)}"),
        ("order agent", bool(LATE_WORDS.search(run.replies[0])),
         f"first reply {'says' if LATE_WORDS.search(run.replies[0]) else 'never says'} the order is late"),
        ("eligibility", bool(eligible),
         f"checks on {ORDER_ID}={len(checks)}, eligible={len(eligible)}"
         + (f"; result: {tool_result(run, checks[-1]['id'])[:120]}" if checks and not eligible else "")),
        ("approval", bool(gated),
         f"approvals shown={[a['rule'] for a in run.approvals]}"),
        ("resolution", resolved, resolution),
    ]
