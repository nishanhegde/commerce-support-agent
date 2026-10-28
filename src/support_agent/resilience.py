"""Day 56: what the graph does when something it depends on fails.

Three layers, each where it can do the least damage:

    call      a tool call fails        retry it, if it is a read (retry_call)
    node      the model call fails     RetryPolicy re-runs the `agent` node
    graph     retries ran out          the node's error_handler decides where
                                       the run goes next (react.py: `recover`)

Why the retry sits in a different place for each:

  - The `agent` node is one model call and nothing else, so re-running the
    node is re-running the call. LangGraph's RetryPolicy does that, with
    backoff.
  - The `tools` node runs a *batch* of calls. A RetryPolicy on it would re-run
    the whole batch, so a create_return that already succeeded would run
    again because the lookup next to it timed out. It's Day 44's "resume
    re-runs the node" trap again. So the retry is per call, inside the node,
    and only for tools marked read-only. A write that failed has an unknown
    outcome, and the safe move is to say so, not to repeat it.

Why timeouts aren't a node setting: add_node(timeout=...) only works on async
nodes ("sync nodes cannot be safely cancelled in-process"), and this graph is
sync. Timeouts live at the call instead: the model client's `timeout`, and
the MCP bridge's CALL_TIMEOUT.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import anthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, ToolException
from langgraph.types import RetryPolicy

T = TypeVar("T")


class ToolUnavailable(ToolException):
    """The system behind a tool isn't answering: timed out, or the connection
    is gone. Different from a tool that ran and said no (plain ToolException):
    that one is an answer, this one is the absence of an answer."""


# What is worth trying the model again for. All three are "not your fault,
# not permanent": a dropped connection (APITimeoutError is a subclass), 429,
# and 5xx (529 "overloaded" included). A 400 or 401 would fail identically
# every time, so retrying it only delays the error.
TRANSIENT_MODEL_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


def is_transient(error: BaseException) -> bool:
    return isinstance(error, (*TRANSIENT_MODEL_ERRORS, ToolUnavailable))


def is_read_only(tool: BaseTool) -> bool:
    """Safe to call twice. Set on the tool's metadata where it is defined (the
    local lookups), or from the server's readOnlyHint (MCP tools)."""
    return bool((tool.metadata or {}).get("read_only"))


def mark_read_only(*tools: BaseTool) -> None:
    for t in tools:
        t.metadata = {**(t.metadata or {}), "read_only": True}


@dataclass(frozen=True)
class Resilience:
    model_attempts: int = 3  # per `agent` run, counting the first
    tool_attempts: int = 2  # per read-only call, counting the first
    base_delay: float = 0.5  # seconds; doubles each retry. 0 in tests.
    # Tried once, after the primary's retries are used up. A smaller model
    # that answers beats a bigger one that doesn't. Tests script it; None
    # means no degraded mode.
    fallback_model: BaseChatModel | None = None

    def node_retry(self) -> RetryPolicy:
        return RetryPolicy(
            initial_interval=self.base_delay,
            max_attempts=self.model_attempts,
            jitter=self.base_delay > 0,
            retry_on=is_transient,
        )


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int,
    base_delay: float,
    retry_on: Callable[[BaseException], bool] = is_transient,
    on_retry: Callable[[int, BaseException], None] = lambda attempt, error: None,
) -> T:
    """Call fn(); on a retryable error wait base_delay * 2**n and try again,
    up to `attempts` calls in all. The last error is re-raised unchanged.

    No jitter: this is one agent's call to one local server, not a fleet
    hammering a shared API. The model retry has it (RetryPolicy, jitter on).
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts or not retry_on(e):
                raise
            on_retry(attempt, e)
            time.sleep(base_delay * 2 ** (attempt - 1))
    raise AssertionError("unreachable")  # the loop returns or raises


APOLOGY = (
    "Sorry, I ran into a technical problem and can't finish this right now. "
    "Please try again in a few minutes. If you were changing an order, check "
    "its status before repeating the request."
)


def recovery_messages(messages: list) -> list:
    """What to append to the history when a run is abandoned.

    If the last model message asked for tools, each call needs a result
    before the next API request, or Claude rejects the whole history with a
    400, and the thread is bricked until someone edits the checkpoint. So
    every call without a result gets one, saying what is true: the outcome
    is unknown (the node failed, and a write inside it may have landed
    before it did).
    """
    out: list = []
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage):
        for c in last.tool_calls:
            out.append(
                ToolMessage(
                    '{"error": "outcome unknown: the system failed while running this call"}',
                    tool_call_id=c["id"],
                    name=c["name"],
                    status="error",
                )
            )
    out.append(AIMessage(APOLOGY))
    return out
