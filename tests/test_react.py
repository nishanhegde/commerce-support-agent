"""Day 42 tests: no API calls. A scripted fake model plays Claude's part."""

import itertools
import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from support_agent.react import build_explicit, build_prebuilt, final_text


class ScriptedModel(GenericFakeChatModel):
    """Replays a fixed sequence of AIMessages. bind_tools is a no-op: the script
    already decides which tools get called."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*responses):
    return ScriptedModel(messages=iter(responses))


def call(name, args, id="c1"):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


def looping_model():
    """A model that asks for a tool forever, the ping-pong case the guards exist for."""
    return ScriptedModel(
        messages=(
            AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1042"}, id=f"c{i}")])
            for i in itertools.count()
        )
    )


def ask(graph, text, **config):
    return graph.invoke({"messages": [("user", text)]}, config=config or None)


def tool_messages(result):
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


# --- explicit graph --------------------------------------------------------


def test_one_tool_round_trip():
    model = scripted(
        AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1042"})]),
        AIMessage("Order 1042 is with BlueDart."),
    )
    out = ask(build_explicit(model), "Where is order 1042?")

    assert final_text(out) == "Order 1042 is with BlueDart."
    assert out["iterations"] == 2
    [tm] = tool_messages(out)
    assert json.loads(tm.content)["carrier"] == "BlueDart"


def test_parallel_tool_calls_each_get_a_result():
    model = scripted(
        AIMessage(
            "",
            tool_calls=[
                call("get_order_status", {"order_id": "1042"}, id="a"),
                call("search_products", {"keywords": ["insole"]}, id="b"),
            ],
        ),
        AIMessage("Shipped, and insoles are in stock."),
    )
    out = ask(build_explicit(model), "Has 1042 shipped, and do you have insoles?")

    assert [tm.tool_call_id for tm in tool_messages(out)] == ["a", "b"]


@pytest.mark.parametrize(
    "bad_call",
    [
        call("cancel_order", {"order_id": "1042"}),  # no such tool
        call("get_order_status", {}),  # missing required argument
    ],
)
def test_tool_errors_go_back_to_the_model_not_up_the_stack(bad_call):
    model = scripted(AIMessage("", tool_calls=[bad_call]), AIMessage("Sorry, try again."))
    out = ask(build_explicit(model), "Cancel 1042")

    [tm] = tool_messages(out)
    assert tm.status == "error"
    assert final_text(out) == "Sorry, try again."


def test_max_iterations_ends_cleanly():
    out = ask(build_explicit(looping_model(), max_iterations=2), "Where is 1042?")

    assert out["iterations"] == 2
    assert final_text(out).startswith("[stopped after 2 iterations")


def test_recursion_limit_is_the_backstop_when_our_guard_is_too_loose():
    # max_iterations=100 is effectively no guard. Each loop is 2 supersteps
    # (agent, tools), so recursion_limit=5 trips on the third agent step.
    graph = build_explicit(looping_model(), max_iterations=100)
    with pytest.raises(GraphRecursionError):
        ask(graph, "Where is 1042?", recursion_limit=5)


# --- prebuilt --------------------------------------------------------------


def test_prebuilt_runs_the_same_loop():
    model = scripted(
        AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1042"})]),
        AIMessage("Order 1042 is with BlueDart."),
    )
    out = ask(build_prebuilt(model), "Where is order 1042?")

    assert final_text(out) == "Order 1042 is with BlueDart."
    assert len(tool_messages(out)) == 1


def test_prebuilt_stops_gracefully_before_the_recursion_limit():
    # The prebuilt tracks `remaining_steps` (a managed value derived from
    # recursion_limit) and, when the model wants another tool with too few
    # steps left, answers instead of raising. Our explicit graph has to get
    # the same effect from max_iterations.
    out = ask(build_prebuilt(looping_model()), "Where is 1042?", recursion_limit=6)

    assert "need more steps" in final_text(out)
