"""Day 51: a supervisor and three workers.

    START -> [load_customer] -> supervisor -+-> order_agent ---+
                                  ^         +-> returns_agent -+--> (back to supervisor)
                                  |         +-> general_agent -+
                                  +---------+
                                            +-> END   (FINISH: reply to the customer)

The supervisor never calls a commerce tool. Each turn it either picks a
worker and writes it a brief, or decides the job is done and writes the
customer's reply from what the workers reported. Workers are Day 42's
ReAct graph (build_explicit), each with a slice of the tools and a narrower
prompt.

Workers are isolated. A worker gets one message, the supervisor's brief,
and returns one, its report. It never sees the conversation, the other
workers, or the customer. That's the point of a supervisor: each worker's
context holds only its job, and only the supervisor holds the whole
picture. It's also the cost: anything the worker needs (an order id, "the
customer has confirmed") must be in the brief, or the worker doesn't know
it. Day 53's shared state is the other choice.

Guard rails survive the split. Each worker is a full build_explicit graph,
so its tools node still runs the Day 49 policy and its `approve` node still
interrupts. That interrupt surfaces on this graph, so chat.py's approval
loop works unchanged. A worker can't get around the gate, and neither can
the supervisor, which has no tools to get around it with.

Routing uses a forced tool call, not free text: the supervisor must call
`Route`, so its decision arrives as validated arguments rather than prose
to parse. This is what with_structured_output does under the hood; it's
written out here so the scripted test model can play it.

Day 52: handoffs. A worker that finds the job belongs to another worker can
pass it straight there, without a round trip through the supervisor:

    supervisor -> order_agent ==handoff==> returns_agent -> supervisor

The handing-off worker calls a `transfer_to_<worker>` tool with a reason
and a context note. That ends its run (stop_after), and its node returns
Command(goto=<worker>) instead of going back up. Command is how a node says
both "here's my state update" and "go there next" in one return value; the
graph has no static edge out of a worker, so every exit is a Command.

Context transfer is the context note, not the transcript. The receiving
worker gets a brief built from the original request, the reason and the
note. That keeps the receiver's context small and on-topic, at the cost of
trusting the sender to write down every fact that matters. The tool's
description says exactly which facts.

Every handoff is a row in the audit log (kind "handoff") and an entry in
state["handoffs"], and the supervisor sees it in the reports. It's also
bounded: at most max_handoffs per turn, and each one counts as a
delegation, so two workers can't pass a case back and forth forever.

Day 53: shared=True switches the workers from isolated to shared state.
Each worker then also reads the whole customer conversation and the case
file (case.py), and writes facts to the case file with update_case, under
that module's conflict rules. The supervisor reads the case file too. Same
graph, same edges: only what each agent can see and write changes.

Day 54: escalation is the supervisor's call, not a worker's tool. Only the
supervisor holds the whole conversation, which is what a handover needs.
Route(next="ESCALATE") goes to the `escalate` node (escalation.py:
summarise, package, transfer). From then on the thread is the human's:
START routes every new message to `with_human`, which adds it to the
ticket and acknowledges it, with no model call.

    START -+-> with_human -> END             (thread already transferred)
           +-> [load_customer] -> supervisor -> ... -> escalate -> END
"""

from pathlib import Path
from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.config import get_config
from langgraph.store.base import BaseStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from support_agent import returns
from support_agent.audit import AuditLog, audited
from support_agent.case import case_block, check_writes, merge_case, update_case_tool, writes_in
from support_agent.escalation import (
    HANDOFF_DIR,
    actions_from_audit,
    add_after_transfer,
    holding_reply,
    next_ticket_id,
    package,
    summarise,
    transcript_lines,
    transfer,
    transferred_reply,
)
from support_agent.memory import CustomerContext, customer_prompt, load_customer, save_customer_note
from support_agent.policy import Policy, default_policy
from support_agent.react import build_explicit, default_model, message_text
from support_agent.returns import check_return_eligibility, create_return
from support_agent.tools import cancel_order, get_order_status, search_products

WorkerName = Literal["order_agent", "returns_agent", "general_agent"]

# Each worker: its tools, and what the supervisor is told it's for. The
# descriptions go into the supervisor's prompt, so they double as routing
# rules. Read-only tools may overlap (returns needs to see the order); each
# write tool belongs to exactly one worker.
WORKERS: dict[str, tuple[list, str]] = {
    "order_agent": (
        [get_order_status, cancel_order],
        "order status, tracking, delivery dates; cancels orders that haven't shipped",
    ),
    "returns_agent": (
        [get_order_status, check_return_eligibility, create_return],
        "return and refund eligibility, including refunds for late shipments; opens returns",
    ),
    # Day 54: escalate_to_human moved out of here. Escalating is the
    # supervisor's decision (Route ESCALATE), since it holds the whole case.
    "general_agent": (
        [search_products, save_customer_note],
        "product questions and stock, remembering customer preferences",
    ),
}

# Day 52: who may hand off to whom. Order work turns into returns work (a
# late order the customer wants refunded) and back (a "return" that's really
# a cancellation, because it hasn't shipped).
HANDOFFS = {"order_agent": ["returns_agent"], "returns_agent": ["order_agent"]}

TRANSFER_PREFIX = "transfer_to_"


class Handoff(BaseModel):
    reason: str = Field(description="Why this belongs to the other worker, in one line.")
    context: str = Field(
        description="Everything the other worker needs, because it won't see your "
        "conversation: order number, SKUs and quantities, amounts, what you found, "
        "and what the customer wants and has already confirmed."
    )


def handoff_tool(target: str) -> StructuredTool:
    """A tool that hands the case to `target`. Running it does nothing but
    acknowledge: the handoff happens when the worker's node sees the call."""

    def transfer(reason: str, context: str) -> dict:
        return {"handed_off_to": target}

    return StructuredTool.from_function(
        transfer,
        name=f"{TRANSFER_PREFIX}{target}",
        description=(
            f"Hand this case straight to the {target}, whose job is: {WORKERS[target][1]}. "
            "Call it when the brief turns out to need that job, after you've done the part "
            "that's yours. Your run ends with this call: don't write a report afterwards."
        ),
        args_schema=Handoff,
    )


WORKER_SYSTEM = """You are the {name} in a customer support team for an online
retailer. Your job: {job}.

You get a brief from the supervisor, not from the customer, and your reply
goes back to the supervisor, who talks to the customer. So:
- Do what the brief asks with your tools. Never guess a fact a tool can give.
- Report back briefly: what you found, what you did (with ids and amounts),
  and anything the customer still has to decide. Plain facts, no greeting.
- If the brief needs something you can't do, say so; don't improvise.{handoff}{shared}"""

SHARED_HINT = """
- The brief includes the customer conversation and the shared case file.
  Don't redo work the case file already records. Record each fact you own
  with update_case as soon as a tool establishes it."""

HANDOFF_HINT = """
- If the work turns out to belong to the {targets}, hand it over with the
  transfer tool rather than reporting back."""

SUPERVISOR_SYSTEM = """You are the supervisor of a customer support team for an
online retailer. You talk to the customer; your workers do the work. You have
no commerce tools yourself.

Your workers:
{workers}

Each turn, call Route once:
- To delegate, set `next` to a worker and write `task`: a brief that stands
  on its own. The worker never sees this conversation, so include the order
  number, what the customer wants, and what they've already confirmed.
- One worker at a time. After each report, decide again: another worker, or
  done.
- When the customer's request is handled, or needs their answer before
  anything else can happen, set `next` to FINISH and write `reply`: the
  message to the customer, built only from the workers' reports. Don't
  state a fact no worker reported, or promise anything no worker did.
- For small talk or a question you can answer from the conversation alone,
  FINISH straight away.
- Workers may hand a case to each other directly; the reports show it. You
  don't need to delegate again what a handoff already covered.
- Set `next` to ESCALATE, with the reason in `task`, when the customer asks
  for a person, disputes an answer you can't change, or needs something no
  worker can do. Not to get round a reviewer's rejection: only if the
  customer then asks for a person. `priority` is "urgent" only for safety,
  suspected fraud, or a repeat complaint.

Be brief with the customer. If they're upset, acknowledge it first."""


class Route(BaseModel):
    """Decide the next step: delegate to one worker, or finish this turn."""

    next: Literal["order_agent", "returns_agent", "general_agent", "FINISH", "ESCALATE"]
    task: str = Field(default="", description="For a worker: a self-contained brief. For ESCALATE: why. Empty when FINISH.")
    reply: str = Field(default="", description="When FINISH: the message to the customer. Empty otherwise.")
    priority: Literal["normal", "urgent"] = Field(default="normal", description="For ESCALATE only.")


class SupervisorState(MessagesState):
    # `messages` holds only what the customer sees: their messages and the
    # supervisor's replies. Workers' tool calls live in their own runs and
    # in the audit log, not in the conversation.
    customer: dict
    # This turn's work, reset by turn_input. Each worker appends its report
    # by returning the whole list (no reducer: one writer at a time).
    reports: list[dict]
    delegations: int
    # Day 52: this turn's handoffs, in order. Reset by turn_input.
    handoffs: list[dict]
    # Day 53 (shared mode): the case file. The reducer is merge_case, so a
    # node returns one agent's *write* and the reducer applies the conflict
    # rules. Not reset per turn: it's the case's record, not this turn's.
    case: Annotated[dict, merge_case]
    # Day 54: set once the case is transferred, {"ticket_id": ...}. After
    # that, the thread belongs to a human.
    escalated: dict
    priority: str
    # The supervisor's last decision, read by the router and the worker.
    next: str
    task: str


def turn_input(text: str) -> dict:
    """A new customer message. Like persistence.turn_input, it resets the
    per-turn counters, which the checkpoint would otherwise carry over."""
    return {"messages": [("user", text)], "reports": [], "delegations": 0, "handoffs": []}


def build_supervisor(
    model: BaseChatModel | None = None,
    worker_models: dict[str, BaseChatModel] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    audit: AuditLog | None = None,
    policy: Policy = default_policy,
    max_delegations: int = 4,
    max_handoffs: int = 2,
    shared: bool = False,
    summary_model: BaseChatModel | None = None,
    handoff_dir: Path | None = HANDOFF_DIR,
):
    """`model` routes and replies; `worker_models` overrides a worker's model
    by name (tests script each one). Every model defaults to default_model().
    Routing is a classification job, so the supervisor is the first place a
    cheaper model would go; it's the same model here until that's measured."""
    audit = audit or AuditLog()
    worker_models = worker_models or {}
    router = (model or default_model()).bind_tools([Route], tool_choice="Route")
    summariser = summary_model or model or default_model()

    workers = {
        name: build_explicit(
            worker_models.get(name),
            tools=[
                *tools,
                *(handoff_tool(t) for t in HANDOFFS.get(name, [])),
                *([update_case_tool(name)] if shared else []),
            ],
            policy=policy,
            system=WORKER_SYSTEM.format(
                name=name,
                job=job,
                handoff=HANDOFF_HINT.format(targets=" or ".join(HANDOFFS[name])) if name in HANDOFFS else "",
                shared=SHARED_HINT if shared else "",
            ),
            audit=audit,
            name=name,
            stop_after=frozenset(f"{TRANSFER_PREFIX}{t}" for t in HANDOFFS.get(name, [])),
            # No checkpointer: a worker called inside a node inherits this
            # graph's, so its interrupts pause and resume with the parent.
            # Each delegation starts it fresh.
        )
        for name, (tools, job) in WORKERS.items()
    }
    roster = "\n".join(f"- {name}: {job}" for name, (_, job) in WORKERS.items())

    def supervisor(state: SupervisorState) -> dict:
        used = state.get("delegations", 0)
        prompt = (
            SUPERVISOR_SYSTEM.format(workers=roster)
            + f"\n\nToday is {returns.today().isoformat()}."
            + customer_prompt(state.get("customer"))
            + _reports_block(state.get("reports", []))
            + (f"\n\n{case_block(state.get('case'))}" if shared else "")
        )
        if used >= max_delegations:
            prompt += f"\n\nYou have used all {max_delegations} delegations for this turn: FINISH now."
        response = router.invoke([SystemMessage(prompt), *state["messages"]])
        route = _parse(response)

        if route.next != "FINISH" and used >= max_delegations:
            # Told to finish and didn't. Same idea as react.py's give_up: end
            # with something the customer can read, not an exception.
            route = Route(next="FINISH", reply="Sorry, I couldn't finish this. Let me get a person to help.")
        if route.next == "FINISH":
            return {"next": "FINISH", "messages": [AIMessage(route.reply or "How else can I help?")]}
        if route.next == "ESCALATE":
            return {"next": "ESCALATE", "task": route.task, "priority": route.priority}
        return {"next": route.next, "task": route.task, "delegations": used + 1}

    def worker_node(name: str):
        graph = workers[name]

        def run(state: SupervisorState) -> Command:
            # The whole interface: a brief in, a report out. The customer is
            # passed so the worker's prompt has their profile, and its
            # notes tool knows who they are (that comes from the runtime).
            brief = _shared_brief(state) if shared else state["task"]
            result = graph.invoke(
                {"messages": [HumanMessage(brief)], "iterations": 0, "customer": state.get("customer", {})}
            )
            # Day 53: this worker's case-file writes, as one write for the
            # reducer. Rejections are logged here; the reducer re-checks.
            case_update = _case_write(name, result["messages"]) if shared else {}
            call = _handoff_call(result["messages"])
            if call is None:
                report = {"worker": name, "task": state["task"], "report": message_text(result["messages"][-1])}
                return Command(goto="supervisor", update={"reports": [*state.get("reports", []), report], **case_update})
            handoff = _hand_off(state, name, call)
            return Command(goto=handoff.goto, update={**handoff.update, **case_update})

        return run

    def _case_write(name: str, messages: list) -> dict:
        writes = writes_in(messages)
        if not writes:
            return {}
        fields = {k: v for w in writes for k, v in w.get("fields", {}).items()}
        notes = [n for w in writes for n in w.get("notes", [])]
        _, rejected = check_writes(name, fields)
        for field, why in rejected.items():
            audit.record("conflict", "rejected", node=name, field=field, value=fields[field], why=why)
        return {"case": {"by": name, "fields": fields, "notes": notes}}

    def _shared_brief(state: SupervisorState) -> str:
        conversation = "\n".join(
            f"{'customer' if m.type == 'human' else 'supervisor'}: {message_text(m)}" for m in state["messages"]
        )
        return (
            f"Supervisor's task: {state['task']}\n\n"
            f"## Conversation with the customer so far\n{conversation}\n\n"
            f"{case_block(state.get('case'))}"
        )

    def _hand_off(state: SupervisorState, name: str, call: dict) -> Command:
        target = call["name"].removeprefix(TRANSFER_PREFIX)
        doc = {"from": name, "to": target, "reason": call["args"]["reason"],
               "context": call["args"]["context"], "task": state["task"]}
        handoffs = state.get("handoffs", [])
        used = state.get("delegations", 0)
        # Logged here, after the worker's run, so it's written once: on an
        # approval resume this node re-runs, but a run that paused never
        # reached this line.
        if len(handoffs) >= max_handoffs or used >= max_delegations:
            audit.record("handoff", "refused", node=name, why="handoff or delegation limit reached", **doc)
            report = f"wanted to hand off to {target} ({doc['reason']}), but this turn's limit is reached. Found: {doc['context']}"
            return Command(goto="supervisor", update={"reports": [*state.get("reports", []),
                                                                 {"worker": name, "task": state["task"], "report": report}]})
        audit.record("handoff", "handed_off", node=name, **doc)
        report = {"worker": name, "task": state["task"],
                  "report": f"handed off to {target}: {doc['reason']}. Found: {doc['context']}"}
        return Command(
            goto=target,
            update={
                "reports": [*state.get("reports", []), report],
                "handoffs": [*handoffs, doc],
                "task": _handoff_brief(doc),
                "delegations": used + 1,
            },
        )

    def escalate(state: SupervisorState) -> dict:
        # Summarise, package, transfer (escalation.py). The facts the summary
        # is checked against go into its prompt too, so it can't drift far.
        thread_id = get_config()["configurable"].get("thread_id", "-")
        actions = actions_from_audit(audit.rows(thread_id))
        transcript = transcript_lines(state["messages"])
        facts = "\n\n".join(filter(None, [
            customer_prompt(state.get("customer")).strip(),
            case_block(state.get("case")) if shared else "",
            "## Actions on record\n" + ("\n".join(f"- {a}" for a in actions) or "none"),
            _reports_block(state.get("reports", [])).strip(),
        ]))
        summary = summarise(summariser, transcript, facts)
        ticket_id = next_ticket_id()
        doc = package(ticket_id, state["task"], state.get("priority", "normal"), state.get("customer"),
                      summary, state.get("case"), actions, transcript)
        transfer(doc, handoff_dir)
        audit.record("escalation", "transferred", node="escalate", ticket_id=ticket_id,
                     reason=state["task"], priority=doc["priority"], sentiment=summary.sentiment)
        return {"messages": [transferred_reply(ticket_id)], "escalated": {"ticket_id": ticket_id}}

    def with_human(state: SupervisorState) -> dict:
        ticket_id = state["escalated"]["ticket_id"]
        add_after_transfer(ticket_id, message_text(state["messages"][-1]), handoff_dir)
        audit.record("escalation", "message_added", node="with_human", ticket_id=ticket_id)
        return {"messages": [holding_reply(ticket_id)]}

    def route_after_supervisor(
        state: SupervisorState,
    ) -> Literal["order_agent", "returns_agent", "general_agent", "escalate", "__end__"]:
        if state["next"] == "FINISH":
            return END
        return "escalate" if state["next"] == "ESCALATE" else state["next"]

    def route_start(state: SupervisorState) -> str:
        if state.get("escalated"):
            return "with_human"
        if store is not None and "customer" not in state:
            return "load_customer"
        return "supervisor"

    g = StateGraph(SupervisorState, context_schema=CustomerContext)

    def add(node: str, fn) -> None:
        g.add_node(node, audited(node, fn, audit))

    add("supervisor", supervisor)
    for name in WORKERS:
        # Day 52: no static edge out of a worker; its Command says where to
        # go. `destinations` only tells the graph drawing which arrows exist.
        g.add_node(name, audited(name, worker_node(name), audit),
                   destinations=("supervisor", *HANDOFFS.get(name, [])))
    add("escalate", escalate)
    add("with_human", with_human)
    g.add_edge("escalate", END)
    g.add_edge("with_human", END)
    starts = ["with_human", "supervisor"]
    if store is not None:
        add("load_customer", load_customer)
        g.add_edge("load_customer", "supervisor")
        starts.append("load_customer")
    # route_start returns a plain str, so the possible targets are listed
    # here: they depend on whether there's a store.
    g.add_conditional_edges(START, route_start, starts)
    g.add_conditional_edges("supervisor", route_after_supervisor)
    return g.compile(checkpointer=checkpointer, store=store)


def _parse(response: AIMessage) -> Route:
    """The forced Route call's arguments. A reply with no tool call at all
    shouldn't happen with tool_choice forced; if it does, treat its text as
    the answer rather than fail the turn."""
    if response.tool_calls:
        return Route(**response.tool_calls[0]["args"])
    return Route(next="FINISH", reply=message_text(response))


def _handoff_call(messages: list) -> dict | None:
    """The transfer call that ended a worker's run, if one did. It's in the
    last AIMessage, because a transfer tool is a stop_after tool."""
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    if last_ai is None:
        return None
    return next((c for c in last_ai.tool_calls if c["name"].startswith(TRANSFER_PREFIX)), None)


def _handoff_brief(doc: dict) -> str:
    """What the receiving worker reads: the context transfer itself."""
    return (
        f"Handed over to you by the {doc['from']}.\n"
        f"Original request: {doc['task']}\n"
        f"Why it's yours: {doc['reason']}\n"
        f"What the {doc['from']} found: {doc['context']}"
    )


def _reports_block(reports: list[dict]) -> str:
    if not reports:
        return ""
    lines = ["", "", "## Work done this turn (workers' reports; the customer hasn't seen them)"]
    for r in reports:
        lines.append(f"- {r['worker']} was asked: {r['task']}\n  reported: {r['report']}")
    return "\n".join(lines)


def render(node: str, delta) -> list[str]:
    """Terminal lines for one streamed step of the supervisor graph."""
    if node == "__interrupt__" or not delta:
        return []
    if node in ("escalate", "with_human"):
        return [f"agent> {message_text(delta['messages'][-1])}"]
    if node == "supervisor":
        if delta.get("next") == "FINISH":
            return [f"agent> {message_text(delta['messages'][-1])}"]
        if delta.get("next") == "ESCALATE":
            return [f"  supervisor -> ESCALATE ({delta['priority']}): {delta['task']}"]
        return [f"  supervisor -> {delta['next']}: {delta['task']}"]
    if node in WORKERS:
        handoffs = delta.get("handoffs") or []
        if handoffs and handoffs[-1]["from"] == node:
            h = handoffs[-1]
            return [f"  {node} => {h['to']} (handoff): {h['reason']}"]
        lines = []
        if delta.get("case"):
            written = delta["case"]
            lines.append(f"  {node} wrote to case: {', '.join(written['fields']) or '-'}"
                         + (f" (+{len(written['notes'])} notes)" if written["notes"] else ""))
        report = delta["reports"][-1]["report"].replace("\n", " ")
        return [*lines, f"  {node} reported: {report[:160]}{'…' if len(report) > 160 else ''}"]
    return []
