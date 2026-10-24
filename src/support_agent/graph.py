"""Day 41: the smallest useful LangGraph — classify -> handle -> respond.

No LLM and no tools yet. The point is the graph mechanics: typed state, nodes
that return partial state updates, and a conditional edge that routes on intent.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

Intent = Literal["order_status", "returns", "other"]


class SupportState(TypedDict, total=False):
    # total=False: every key is optional, because the graph starts with only
    # `message` and each node fills in more as it runs.
    message: str
    intent: Intent
    answer: str
    reply: str


_ORDER_WORDS = ("order", "where is", "tracking", "shipped", "delivery")
_RETURN_WORDS = ("return", "refund", "exchange", "send back")


def classify(state: SupportState) -> dict:
    text = state["message"].lower()
    if any(w in text for w in _RETURN_WORDS):  # returns first: "refund my order"
        intent: Intent = "returns"
    elif any(w in text for w in _ORDER_WORDS):
        intent = "order_status"
    else:
        intent = "other"
    return {"intent": intent}


def handle_order(state: SupportState) -> dict:
    return {"answer": "Looking up your order status."}


def handle_returns(state: SupportState) -> dict:
    return {"answer": "Starting a return request."}


def handle_other(state: SupportState) -> dict:
    return {"answer": "I can help with orders and returns. Which one do you need?"}


def respond(state: SupportState) -> dict:
    return {"reply": f"[{state['intent']}] {state['answer']}"}


def route_by_intent(state: SupportState) -> str:
    """Router for the conditional edge: returns the *name of the next node*."""
    return {
        "order_status": "handle_order",
        "returns": "handle_returns",
        "other": "handle_other",
    }[state["intent"]]


def build_graph():
    g = StateGraph(SupportState)
    g.add_node("classify", classify)
    g.add_node("handle_order", handle_order)
    g.add_node("handle_returns", handle_returns)
    g.add_node("handle_other", handle_other)
    g.add_node("respond", respond)

    g.add_edge(START, "classify")
    # path_map makes the possible targets explicit so the drawn graph shows
    # all three branches (otherwise LangGraph can't infer them from a function).
    g.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "handle_order": "handle_order",
            "handle_returns": "handle_returns",
            "handle_other": "handle_other",
        },
    )
    for handler in ("handle_order", "handle_returns", "handle_other"):
        g.add_edge(handler, "respond")
    g.add_edge("respond", END)
    return g.compile()
