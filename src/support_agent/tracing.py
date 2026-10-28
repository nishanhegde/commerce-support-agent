"""Day 57: run logs. What a conversation cost, and where the time went.

The audit log (audit.py) answers "what ran, and who allowed it". It is a
record for people who weren't there. This answers a different question, for
the person debugging or paying: how many model calls, how many tokens, how
many rupees' worth of dollars, and which node or tool was slow.

It is a LangChain callback handler. LangGraph runs every node, model call and
tool call inside the callback system, so one handler passed in the config
sees all of them, workers inside a supervisor included, with no change to any
node:

    trace = RunTrace("t1", path=Path("data/runs/t1.jsonl"))
    graph.invoke(turn_input("hi"), thread("t1", trace))
    print(format_summary(trace.summary()))

Events are appended to a JSONL file as they happen, so a run that crashes
still leaves its log. One line per event, three kinds:

    node    a graph node ran:     name, ms, ok
    model   a model call:         model, tokens in/out/cached, ms, cost_usd, ok
    tool    a tool call:          name, ms, status

This is the "clean logs now" version. Month 4 puts real tracing (spans,
a UI, dataset capture) on top of the same events.
"""

import json
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

# Dollars per million tokens, from Month 1's cost.py. Matched by prefix, so a
# dated model id still finds its row. A model that isn't here is logged with
# cost_usd=None and counted as "unpriced" in the summary: a wrong price would
# be worse than a missing one.
PRICE_PER_MILLION = {
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},
}


def cost_usd(model: str | None, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float | None:
    """Dollar cost of one call, or None if the model has no price here.

    LangChain's input_tokens is the TOTAL input, cached tokens included, so
    the uncached part is what's left after taking them out. Billing them all
    at the full rate would overstate any call that hit the cache.
    """
    prices = next((p for name, p in PRICE_PER_MILLION.items() if model and model.startswith(name)), None)
    if prices is None:
        return None
    uncached = max(input_tokens - cache_read - cache_write, 0)
    return (
        uncached * prices["input"]
        + output_tokens * prices["output"]
        + cache_read * prices["cache_read"]
        + cache_write * prices["cache_write"]
    ) / 1_000_000


def node_label(metadata: dict) -> str:
    """'returns_agent/tools' for a node inside a worker, 'tools' at the top.
    langgraph_checkpoint_ns is 'returns_agent:<task id>|tools:<task id>'."""
    ns = metadata.get("langgraph_checkpoint_ns", "")
    return "/".join(part.split(":")[0] for part in ns.split("|") if part) or metadata.get("langgraph_node", "?")


class RunTrace(BaseCallbackHandler):
    """Collects events for one conversation. Thread-safe: parallel tool calls
    and map-reduce branches report from different threads."""

    def __init__(self, thread_id: str, path: Path | None = None):
        self.thread_id = thread_id
        self.path = path
        self.events: list[dict] = []
        self._open: dict[UUID, dict] = {}  # run_id -> what we knew at the start
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    # --- plumbing ----------------------------------------------------------

    def _start(self, run_id: UUID, **info) -> None:
        with self._lock:
            self._open[run_id] = {"t0": time.perf_counter(), **info}

    def _finish(self, run_id: UUID, kind: str, **fields) -> None:
        with self._lock:
            started = self._open.pop(run_id, None)
        if started is None:  # a run we chose not to track
            return
        event = {
            "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "thread_id": self.thread_id,
            "kind": kind,
            "name": started["name"],
            "ms": round((time.perf_counter() - started["t0"]) * 1000, 1),
            **fields,
        }
        with self._lock:
            self.events.append(event)
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")

    # --- nodes -------------------------------------------------------------

    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, name=None, **kwargs):
        # Every runnable inside a node inherits its langgraph_node metadata, so
        # that alone matches dozens of runs. The node itself is the one whose
        # own name equals it.
        if metadata and name and metadata.get("langgraph_node") == name:
            self._start(run_id, name=node_label(metadata))

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._finish(run_id, "node", ok=True)

    def on_chain_error(self, error, *, run_id, **kwargs):
        # A pause for approval is GraphInterrupt: not a failure, so ok=True.
        self._finish(run_id, "node", ok=type(error).__name__ == "GraphInterrupt")

    # --- model calls -------------------------------------------------------

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        self._start(run_id, name="model", ls_model=(metadata or {}).get("ls_model_name"))

    def on_llm_end(self, response, *, run_id, **kwargs):
        message = response.generations[0][0].message
        usage = getattr(message, "usage_metadata", None) or {}
        details = usage.get("input_token_details") or {}
        started = self._open.get(run_id, {})
        model = (message.response_metadata or {}).get("model_name") or started.get("ls_model")
        tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        read, write = details.get("cache_read", 0), details.get("cache_creation", 0)
        self._finish(
            run_id, "model", model=model, input_tokens=tokens_in, output_tokens=tokens_out,
            cache_read=read, cache_write=write, cost_usd=cost_usd(model, tokens_in, tokens_out, read, write), ok=True,
        )

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, "model", error=repr(error), ok=False, cost_usd=0.0)

    # --- tools -------------------------------------------------------------

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._start(run_id, name=kwargs.get("name") or serialized.get("name", "?"))

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._finish(run_id, "tool", status=getattr(output, "status", "success"))

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, "tool", status="error", error=repr(error))

    # --- the per-conversation view ----------------------------------------

    def summary(self) -> dict:
        models = [e for e in self.events if e["kind"] == "model"]
        tools = [e for e in self.events if e["kind"] == "tool"]
        nodes: dict[str, dict] = defaultdict(lambda: {"runs": 0, "ms": 0.0})
        for e in self.events:
            if e["kind"] == "node":
                nodes[e["name"]]["runs"] += 1
                nodes[e["name"]]["ms"] = round(nodes[e["name"]]["ms"] + e["ms"], 1)
        priced = [e["cost_usd"] for e in models if e["cost_usd"] is not None]
        return {
            "thread_id": self.thread_id,
            "model_calls": len(models),
            "model_errors": sum(not e["ok"] for e in models),
            "input_tokens": sum(e.get("input_tokens", 0) for e in models),
            "output_tokens": sum(e.get("output_tokens", 0) for e in models),
            "cost_usd": round(sum(priced), 6),
            "unpriced_calls": len(models) - len(priced),
            "model_ms": round(sum(e["ms"] for e in models), 1),
            "tool_calls": len(tools),
            "tool_errors": sum(e.get("status") == "error" for e in tools),
            "tool_ms": round(sum(e["ms"] for e in tools), 1),
            "nodes": dict(nodes),
        }


def format_summary(s: dict) -> str:
    cost = f"${s['cost_usd']:.4f}" + (f" (+{s['unpriced_calls']} unpriced calls)" if s["unpriced_calls"] else "")
    lines = [
        f"thread {s['thread_id']}",
        f"  model   {s['model_calls']} calls ({s['model_errors']} failed)  {s['input_tokens']} in / {s['output_tokens']} out tokens  {cost}  {s['model_ms'] / 1000:.1f}s",
        f"  tools   {s['tool_calls']} calls ({s['tool_errors']} failed)  {s['tool_ms'] / 1000:.2f}s",
    ]
    for node, n in sorted(s["nodes"].items(), key=lambda kv: -kv[1]["ms"]):
        lines.append(f"  node    {node:<22} {n['runs']:>3} runs  {n['ms'] / 1000:>6.2f}s")
    return "\n".join(lines)
