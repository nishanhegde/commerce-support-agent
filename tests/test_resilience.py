"""Day 56 tests: retries, fallback and recovery. Offline: a scripted model
that fails on demand, and tools that fail on demand."""

import anthropic
import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from support_agent.audit import AuditLog
from support_agent.persistence import thread, turn_input
from support_agent.react import build_explicit, final_text
from support_agent.resilience import (
    APOLOGY, Resilience, ToolUnavailable, is_transient, mark_read_only, retry_call,
)
from test_react import ScriptedModel, call, scripted

# No waiting between attempts in tests.
FAST = dict(base_delay=0)


def connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))


class FlakyModel(ScriptedModel):
    """Raises `error` for its first `failures` calls, then plays its script.
    `calls` counts every attempt, failed or not."""

    failures: int = 0
    error: Exception | None = None
    calls: int = 0

    def _generate(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return super()._generate(*args, **kwargs)


def flaky(failures, *responses, error=None):
    return FlakyModel(messages=iter(responses), failures=failures, error=error or connection_error())


@pytest.fixture
def audit():
    with AuditLog() as log:
        yield log


def run(model, audit, text="hi", resilience=None, **kw):
    graph = build_explicit(
        model, checkpointer=InMemorySaver(), audit=audit, resilience=resilience or Resilience(**FAST), **kw
    )
    return graph, graph.invoke(turn_input(text), thread("t"))


def outcomes(audit, kind="recovery"):
    return [r["outcome"] for r in audit.rows("t", kind)]


# --- the model -------------------------------------------------------------


def test_a_transient_model_error_is_retried_by_the_node(audit):
    model = flaky(2, AIMessage("Hello."))
    _, out = run(model, audit)

    assert final_text(out) == "Hello."
    assert model.calls == 3
    # Each failed attempt is a failed transition: the retries show in the log.
    agent_rows = [r["outcome"] for r in audit.rows("t", "transition") if r["node"] == "agent"]
    assert agent_rows == ["failed", "failed", "completed"]
    assert outcomes(audit) == []  # retried is not recovered: nothing went wrong in the end


def test_when_retries_run_out_the_run_ends_with_a_reply_not_an_exception(audit):
    model = flaky(99, AIMessage("never reached"))
    _, out = run(model, audit)

    assert model.calls == 3  # Resilience.model_attempts
    assert final_text(out) == APOLOGY
    assert outcomes(audit) == ["gave_up"]


def test_a_fallback_model_answers_when_the_primary_is_down(audit):
    primary = flaky(99, AIMessage("never reached"))
    backup = scripted(AIMessage("Backup here."))
    _, out = run(primary, audit, resilience=Resilience(fallback_model=backup, **FAST))

    assert final_text(out) == "Backup here."
    assert outcomes(audit) == ["fell_back"]


def test_the_fallbacks_answer_is_routed_like_any_other(audit):
    # The backup asks for a tool: the run must go on to `tools`, not stop.
    backup = scripted(
        AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1042"})]),
        AIMessage("Shipped."),
    )
    _, out = run(flaky(99), audit, resilience=Resilience(fallback_model=backup, **FAST))

    assert [m.name for m in out["messages"] if isinstance(m, ToolMessage)] == ["get_order_status"]
    assert final_text(out) == "Shipped."


def test_a_broken_fallback_ends_in_the_apology_too(audit):
    _, out = run(flaky(99), audit, resilience=Resilience(fallback_model=flaky(99), **FAST))

    assert final_text(out) == APOLOGY
    assert outcomes(audit) == ["fallback_failed", "gave_up"]


def test_a_permanent_error_is_not_retried(audit):
    model = flaky(99, AIMessage("never reached"), error=ValueError("bad request"))
    _, out = run(model, audit)

    assert model.calls == 1
    assert final_text(out) == APOLOGY


def test_which_errors_count_as_transient():
    assert is_transient(connection_error())
    assert is_transient(anthropic.APITimeoutError(request=httpx.Request("POST", "https://x")))
    assert is_transient(ToolUnavailable("down"))
    assert not is_transient(ValueError("bad"))


# --- the tools -------------------------------------------------------------


def make_tool(name, *, fails, read_only):
    """A tool that raises ToolUnavailable `fails` times, then answers.
    Returns it with its call counter (a Pydantic tool won't take new attributes)."""
    state = {"calls": 0}

    @tool(name, description="test tool")
    def t(order_id: str) -> dict:
        state["calls"] += 1
        if state["calls"] <= fails:
            raise ToolUnavailable("connection closed")
        return {"ok": True}

    if read_only:
        mark_read_only(t)
    return t, state


def test_a_read_that_cannot_be_reached_is_retried(audit):
    lookup, n = make_tool("lookup", fails=1, read_only=True)
    model = scripted(
        AIMessage("", tool_calls=[call("lookup", {"order_id": "1"})]), AIMessage("Done.")
    )
    _, out = run(model, audit, tools=[lookup])

    assert n["calls"] == 2
    [tm] = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tm.status == "success"
    assert outcomes(audit) == ["retried"]


def test_a_write_that_cannot_be_reached_is_never_retried(audit):
    # "No answer" doesn't mean "didn't happen", so repeating it could do it twice.
    write, n = make_tool("write", fails=1, read_only=False)
    model = scripted(
        AIMessage("", tool_calls=[call("write", {"order_id": "1"})]), AIMessage("Can't right now.")
    )
    _, out = run(model, audit, tools=[write])

    assert n["calls"] == 1
    [tm] = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tm.status == "error" and "unavailable" in tm.content
    assert final_text(out) == "Can't right now."  # the model got to answer


def test_a_read_that_stays_down_ends_as_an_error_the_model_can_read(audit):
    lookup, n = make_tool("lookup", fails=99, read_only=True)
    model = scripted(
        AIMessage("", tool_calls=[call("lookup", {"order_id": "1"})]), AIMessage("Try later.")
    )
    _, out = run(model, audit, tools=[lookup])

    assert n["calls"] == 2  # Resilience.tool_attempts
    [tm] = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tm.status == "error" and "Don't call it again" in tm.content
    assert final_text(out) == "Try later."


def test_an_unexpected_tool_crash_leaves_a_thread_that_can_continue(audit):
    @tool
    def broken(order_id: str) -> dict:
        """Always crashes."""
        raise RuntimeError("boom")

    model = scripted(
        AIMessage("", tool_calls=[call("broken", {"order_id": "1"}, id="a"), call("broken", {"order_id": "2"}, id="b")])
    )
    graph, out = run(model, audit, tools=[broken])

    assert final_text(out) == APOLOGY
    # Every tool_use has its tool_result: without them the next API call is a 400.
    answered = {m.tool_call_id for m in out["messages"] if isinstance(m, ToolMessage)}
    assert answered == {"a", "b"}
    assert outcomes(audit) == ["gave_up"]


def test_recovery_rows_print(audit):
    from support_agent.audit import format_row

    run(flaky(99), audit)
    lines = [format_row(r) for r in audit.rows("t", "recovery")]
    assert len(lines) == 1 and "RECOVERY gave_up" in lines[0]
