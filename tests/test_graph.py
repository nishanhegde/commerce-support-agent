import pytest

from support_agent.graph import build_graph


@pytest.fixture(scope="module")
def graph():
    return build_graph()


@pytest.mark.parametrize(
    "message, intent",
    [
        ("Where is my order 12345?", "order_status"),
        ("I want a refund", "returns"),
        ("refund my order please", "returns"),  # both keyword sets hit; returns wins
        ("do you sell gift cards", "other"),
    ],
)
def test_routes_by_intent(graph, message, intent):
    out = graph.invoke({"message": message})
    assert out["intent"] == intent
    assert out["reply"].startswith(f"[{intent}]")


def test_final_state_keeps_every_key(graph):
    out = graph.invoke({"message": "track my order"})
    assert set(out) == {"message", "intent", "answer", "reply"}


def test_graph_has_all_branches(graph):
    nodes = set(graph.get_graph().nodes)
    assert {"classify", "handle_order", "handle_returns", "handle_other", "respond"} <= nodes
