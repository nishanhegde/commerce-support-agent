"""Day 49: the audit trail. One append-only table, two kinds of row:

    transition   a node ran: which one, at which step, and what it changed
    approval     an action was approved or rejected, by whom, under what rule

Why not just use the checkpoints? They already hold every step's state, but
they are built for resuming, not for auditing: a fork adds a branch,
update_state() rewrites history from any point, and a deleted thread takes
its checkpoints with it. An audit log has to answer "what actually ran, and
who allowed it" even then, so it is a separate store that is only ever
appended to. Same split as order history vs. the order's audit table in a
commerce platform.

Transitions are recorded by wrapping each node (`audited`, used in react.py),
so nothing inside a node has to remember to log. The wrapper sees things a
node can't, and records them:

  - a pause: interrupt() raises GraphInterrupt, which passes through here
  - the re-run on resume: the same node at the same step, logged twice,
    first "paused", then "completed". That's Day 44's "resume re-runs the
    node" made visible.
  - a failure: any other exception, logged and re-raised
"""

import functools
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_config
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

AUDIT_PATH = Path(__file__).resolve().parents[2] / "data" / "audit.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    at        TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    kind      TEXT NOT NULL,   -- 'transition' | 'approval' | 'handoff' (Day 52) | 'conflict' (Day 53) | 'escalation' (Day 54)
    step      INTEGER,
    node      TEXT,
    outcome   TEXT NOT NULL,   -- completed/paused/failed | auto_approved/approved/rejected
    detail    TEXT NOT NULL    -- JSON
)"""


class AuditLog:
    """Append-only. There's a write method and a read method, and no update
    or delete, which is the whole point.

    AuditLog() with no path is an in-memory database, for tests and for
    graphs built without one.
    """

    def __init__(self, path: Path | str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(exist_ok=True)
        # check_same_thread=False: LangGraph may run a node on a worker
        # thread. The lock makes the writes take turns.
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.execute(SCHEMA)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    # Context manager methods, so `with AuditLog(path) as audit:` closes it.
    # The class-based alternative to @contextmanager (persistence.py).
    def __enter__(self) -> "AuditLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def record(self, kind: str, outcome: str, *, step: int | None = None, node: str | None = None, **detail) -> None:
        # Everything after the bare `*` is keyword-only: record("approval",
        # "approved", node="approve", ...). Java has nothing like it; it
        # stops a caller mixing up `step` and `node` by position.
        thread_id = _current_thread_id()
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit (at, thread_id, kind, step, node, outcome, detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    thread_id,
                    kind,
                    step,
                    node,
                    outcome,
                    json.dumps(detail, ensure_ascii=False, default=str),
                ),
            )

    def rows(self, thread_id: str | None = None, kind: str | None = None) -> list[dict]:
        sql, params = "SELECT * FROM audit WHERE 1=1", []
        if thread_id is not None:
            sql, params = sql + " AND thread_id = ?", [*params, thread_id]
        if kind is not None:
            sql, params = sql + " AND kind = ?", [*params, kind]
        cur = self._conn.execute(sql + " ORDER BY id", params)
        cols = [c[0] for c in cur.description]
        return [
            {**row, "detail": json.loads(row["detail"])}
            for row in (dict(zip(cols, r)) for r in cur.fetchall())
        ]


def _current_thread_id() -> str:
    """The thread being run, from the config LangGraph sets for each node.
    "-" outside a run, or for a graph without a checkpointer."""
    try:
        return get_config().get("configurable", {}).get("thread_id", "-")
    except RuntimeError:  # get_config() outside a running graph
        return "-"


def audited(name: str, fn, log: AuditLog):
    """Wrap a node so every run of it leaves a transition row.

    @functools.wraps copies fn's name and signature onto the wrapper, and
    LangGraph reads the signature to decide what to inject. So a node that
    declares `runtime` still gets it, through **kwargs, as if unwrapped.
    """

    @functools.wraps(fn)
    def node(state, **kwargs):
        step = get_config().get("metadata", {}).get("langgraph_step")
        try:
            update = fn(state, **kwargs)
        except GraphInterrupt:
            log.record("transition", "paused", step=step, node=name, waiting_for="human approval")
            raise
        except Exception as e:
            log.record("transition", "failed", step=step, node=name, error=repr(e))
            raise
        log.record("transition", "completed", step=step, node=name, **summarise(update))
        return update

    return node


def summarise(update: dict | Command | None) -> dict:
    """What a node changed, small enough to read in a log. Full state is in
    the checkpoints; the log says what moved.

    Day 52: a node may return a Command (an update plus where to go next)
    instead of a plain dict. The update is summarised the same way, and
    `goto` is recorded: for a worker, that's the handoff or the return to
    the supervisor, on the record."""
    if isinstance(update, Command):
        return {**summarise(update.update), "goto": update.goto}
    if not update:
        return {"changed": []}
    out: dict = {"changed": sorted(update)}
    for m in update.get("messages", []):
        if isinstance(m, AIMessage) and m.tool_calls:
            out.setdefault("tool_calls", []).extend(
                {"id": c["id"], "name": c["name"], "args": c["args"]} for c in m.tool_calls
            )
        elif isinstance(m, ToolMessage):
            out.setdefault("tool_results", []).append({"id": m.tool_call_id, "name": m.name, "status": m.status})
        elif isinstance(m, AIMessage):
            out["answer_chars"] = len(str(m.content))
    if "customer" in update:
        out["customer_id"] = update["customer"].get("customer_id")
    if update.get("rejected"):
        out["rejected"] = sorted(update["rejected"])
    if "next" in update:  # Day 51: the supervisor's routing decision
        out["next"] = update["next"]
    return out


def format_row(row: dict) -> str:
    """One audit row as one terminal line."""
    d = row["detail"]
    at = row["at"][11:23]  # time of day is enough on screen; the row has the full stamp
    if row["kind"] == "escalation":
        extra = f"  ({d['priority']}, customer {d['sentiment']}): {d['reason']}" if row["outcome"] == "transferred" else ""
        return f"{at}  {row['thread_id']:<10} ESCALATE {row['outcome']:<13} {d['ticket_id']}{extra}"
    if row["kind"] == "conflict":
        return f"{at}  {row['thread_id']:<10} CONFLICT {row['outcome']:<13} {row['node']} wrote {d['field']}={json.dumps(d['value'], ensure_ascii=False)}  ({d['why']})"
    if row["kind"] == "handoff":
        return f"{at}  {row['thread_id']:<10} HANDOFF  {row['outcome']:<13} {d['from']} => {d['to']}  ({d['reason']})"
    if row["kind"] == "approval":
        who = d["by"] if d["by"] != "policy" else f"policy:{d['rule']}"
        extra = f"  reason: {d['reason']}" if d.get("reason") else ""
        return f"{at}  {row['thread_id']:<10} APPROVAL {row['outcome']:<13} {d['tool']}({json.dumps(d['args'])})  by {who}  ({d['why']}){extra}"
    what = ""
    if "tool_calls" in d:
        what = "-> " + ", ".join(c["name"] for c in d["tool_calls"])
    elif "tool_results" in d:
        what = "<- " + ", ".join(f"{r['name']}[{r['status']}]" for r in d["tool_results"])
    elif "answer_chars" in d:
        what = f"answer ({d['answer_chars']} chars)"
    elif "customer_id" in d:
        what = f"customer {d['customer_id']}"
    elif "waiting_for" in d:
        what = f"waiting for {d['waiting_for']}"
    elif "error" in d:
        what = d["error"]
    elif "next" in d:
        what = f"route -> {d['next']}"
    elif "goto" in d:
        what = f"-> {d['goto']}"
    return f"{at}  {row['thread_id']:<10} step {row['step']:>2}  {row['node']:<13} {row['outcome']:<9} {what}"
