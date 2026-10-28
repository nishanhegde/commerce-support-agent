"""Day 56 chaos tests: kill the real mcp-commerce mid-conversation.

Each test starts its own server (the shared one in test_mcp_bridge would be
dead for everyone after the first kill). The model is scripted; nothing here
calls the API.
"""

import os
import shutil
import signal
import subprocess

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from support_agent.audit import AuditLog
from support_agent.mcp_bridge import DEFAULT_SERVER_DIR, MCPBridge, mcp_commerce_server
from support_agent.persistence import thread, turn_input
from support_agent.policy import by_name
from support_agent.react import build_explicit, final_text
from support_agent.resilience import Resilience, ToolUnavailable
from test_react import call, scripted

pytestmark = pytest.mark.skipif(
    not DEFAULT_SERVER_DIR.exists() or shutil.which("uv") is None,
    reason="needs ../mcp-commerce and uv",
)

SEARCH = {"query": "shoe"}


def descendants(pid: int) -> list[int]:
    out = []
    for child in subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split():
        out += [int(child), *descendants(int(child))]
    return out


def kill_server() -> None:
    """SIGKILL everything this process started: `uv run`, and the server under it.
    A kill, not a shutdown: the server gets no chance to say goodbye."""
    for pid in descendants(os.getpid()):
        os.kill(pid, signal.SIGKILL)


@pytest.fixture
def bridge():
    with MCPBridge(mcp_commerce_server()) as b:
        yield b


def test_a_dead_server_is_unavailable_not_a_crash(bridge):
    assert "SH-204" in bridge.call("search_products", SEARCH)
    kill_server()

    with pytest.raises(ToolUnavailable, match="Connection closed"):
        bridge.call("search_products", SEARCH)
    with pytest.raises(ToolUnavailable):  # and stays that way, immediately
        bridge.call("get_order_status", {"order_id": "1042"})


def test_a_slow_server_times_out_instead_of_hanging(bridge):
    bridge._call_timeout = 0.0001  # shorter than any round trip

    with pytest.raises(ToolUnavailable, match="no answer"):
        bridge.call("search_products", SEARCH)


def test_the_conversation_survives_the_server_dying_mid_thread(bridge):
    audit = AuditLog()
    graph = build_explicit(
        scripted(
            # turn 1: works
            AIMessage("", tool_calls=[call("search_products", SEARCH, id="a")]),
            AIMessage("Found shoes."),
            # turn 2: the server is gone
            AIMessage("", tool_calls=[call("search_products", SEARCH, id="b")]),
            AIMessage("I can't look that up right now."),
            # turn 3: the customer is still there, and the thread still works
            AIMessage("Anything else I can do?"),
        ),
        checkpointer=InMemorySaver(),
        tools=bridge.tools,
        policy=by_name(bridge.needs_approval),
        audit=audit,
        resilience=Resilience(base_delay=0),
    )
    t = thread("chaos")

    assert final_text(graph.invoke(turn_input("shoes?"), t)) == "Found shoes."

    kill_server()
    out = graph.invoke(turn_input("and sandals?"), t)

    [failed] = [m for m in out["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "b"]
    assert failed.status == "error" and "unavailable" in failed.content
    assert final_text(out) == "I can't look that up right now."
    # A read, so it was tried twice before giving up.
    assert [r["outcome"] for r in audit.rows("chaos", "recovery")] == ["retried"]

    assert final_text(graph.invoke(turn_input("thanks"), t)) == "Anything else I can do?"
    audit.close()


def test_a_write_to_a_dead_server_is_not_repeated(bridge):
    audit = AuditLog()
    graph = build_explicit(
        scripted(
            AIMessage("", tool_calls=[call("create_cart", {}, id="w")]),
            AIMessage("I couldn't start a cart."),
        ),
        checkpointer=InMemorySaver(),
        tools=bridge.tools,
        policy=lambda c: None,  # no human in this test: the point is the retry rule
        audit=audit,
        resilience=Resilience(base_delay=0),
    )
    kill_server()
    out = graph.invoke(turn_input("start a cart"), thread("w"))

    assert final_text(out) == "I couldn't start a cart."
    assert audit.rows("w", "recovery") == []  # no retry rows: one attempt only
    audit.close()
