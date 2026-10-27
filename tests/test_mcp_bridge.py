"""Day 46 tests: mcp-commerce's tools inside the graph.

Integration tests: they start the real mcp-commerce server (stdio, OCC mock)
once for the module. The model is still scripted, so no API calls. Skipped
if the sibling repo or `uv` isn't there.
"""

import json
import shutil

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException

from support_agent.hitl import decision, pending_approval
from support_agent.mcp_bridge import DEFAULT_SERVER_DIR, MCPBridge, mcp_commerce_server
from support_agent.persistence import sqlite_checkpointer, thread, turn_input
from support_agent.policy import by_name
from support_agent.react import build_explicit, final_text
from test_react import call, scripted

pytestmark = pytest.mark.skipif(
    not DEFAULT_SERVER_DIR.exists() or shutil.which("uv") is None,
    reason="needs ../mcp-commerce and uv",
)


@pytest.fixture(scope="module")
def bridge():
    # Module scope: one server process for every test here, ~1s instead of
    # ~1s per test. The tests below only read, or write carts they own.
    with MCPBridge(mcp_commerce_server()) as b:
        yield b


def mcp_graph(bridge, *responses, checkpointer=None):
    return build_explicit(
        scripted(*responses),
        checkpointer=checkpointer,
        tools=bridge.tools,
        policy=by_name(bridge.needs_approval),
    )


# --- the bridge ------------------------------------------------------------


def test_discovers_all_seven_tools(bridge):
    assert {t.name for t in bridge.tools} == {
        "search_products", "get_product", "create_cart", "get_cart",
        "add_to_cart", "get_order_history", "get_order_status",
    }


def test_approval_set_comes_from_read_only_annotations(bridge):
    assert bridge.needs_approval == {"create_cart", "add_to_cart"}


def test_server_schema_reaches_the_model_unchanged(bridge):
    [search] = [t for t in bridge.tools if t.name == "search_products"]
    page_size = search.tool_call_schema["properties"]["page_size"]
    assert (page_size["minimum"], page_size["maximum"]) == (1, 20)


def test_call_returns_the_servers_json_text(bridge):
    order = json.loads(bridge.call("get_order_status", {"order_code": "1042"}))
    assert order["shipments"][0]["carrier"] == "BlueDart"


def test_server_errors_become_tool_exceptions(bridge):
    with pytest.raises(ToolException, match="not found"):
        bridge.call("get_order_status", {"order_code": "nope"})


# --- in the graph ------------------------------------------------------------


def test_wismo_chain_runs_and_survives_a_restart(bridge, tmp_path):
    db = tmp_path / "cp.sqlite"
    with sqlite_checkpointer(db) as saver:
        graph = mcp_graph(
            bridge,
            AIMessage("", tool_calls=[call("get_order_history", {}, id="h")]),
            AIMessage("", tool_calls=[call("get_order_status", {"order_code": "1042"}, id="s")]),
            AIMessage("It's with BlueDart, and late."),
            checkpointer=saver,
        )
        out = graph.invoke(turn_input("where are my shoes?"), thread("t"))

    results = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.status for m in results] == ["success", "success"]
    # MCP's text went in as-is: valid JSON, not a JSON-encoded string.
    assert json.loads(results[1].content)["order_code"] == "1042"

    with sqlite_checkpointer(db) as saver:
        graph = mcp_graph(bridge, checkpointer=saver)
        state = graph.get_state(thread("t")).values
    assert final_text(state) == "It's with BlueDart, and late."


def test_a_server_error_goes_back_to_the_model(bridge):
    graph = mcp_graph(
        bridge,
        AIMessage("", tool_calls=[call("get_order_status", {"order_code": "9999"})]),
        AIMessage("I couldn't find that order."),
    )
    out = graph.invoke(turn_input("where is 9999?"))
    [tm] = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tm.status == "error"
    assert "not found" in tm.content


def test_a_write_tool_waits_for_approval(bridge, tmp_path):
    with sqlite_checkpointer(tmp_path / "cp.sqlite") as saver:
        graph = mcp_graph(
            bridge,
            AIMessage("", tool_calls=[call("create_cart", {})]),
            AIMessage("Cart ready."),
            checkpointer=saver,
        )
        graph.invoke(turn_input("start a cart"), thread("t"))
        [pending] = pending_approval(graph, thread("t"))
        assert pending["name"] == "create_cart"

        out = graph.invoke(decision(True), thread("t"))

    [tm] = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tm.status == "success"
    assert "cart_id" in json.loads(tm.content)
