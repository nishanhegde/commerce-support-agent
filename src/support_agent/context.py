"""Day 45: a reusable subgraph, "look up customer context for one order".

Three lookups (order, shipment, payment) don't depend on each other, so they
run in parallel, then one node merges them:

            +--> order ----+
    START --+--> shipment -+--> merge --> END
            +--> payment --+

Parallel here is LangGraph's *superstep*: every node whose inputs are ready
runs in the same step, and sync nodes in a step run on a thread pool. The
three lookups share a step; `merge` waits for the next one. No threads or
futures in our code, just three edges out of START.

Reusable means it has an interface. ContextInput is what callers must pass,
ContextOutput is all they get back; `order`, `shipment` and `payment` are
internal and never leak into the parent's state. Java analogy: a class with
a public method signature and private fields.

Data is in-memory, like tools.py. Day 46 replaces it with mcp-commerce.
"""

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from support_agent.tools import ORDERS

SHIPMENTS = {
    "1042": {
        "carrier": "BlueDart",
        "last_scan": "Arrived at Bengaluru hub",
        "last_scan_at": "2026-10-26T06:40",
        "estimated_delivery": "2026-10-28",
    },
    "1043": {
        "carrier": "Delhivery",
        "last_scan": "Delivered, signed by: R. Hegde",
        "last_scan_at": "2026-10-20T14:12",
    },
    # 1044 is pending: nothing has shipped, so there is no shipment record.
}

PAYMENTS = {
    "1042": {"method": "UPI", "amount": 7999, "status": "captured"},
    "1043": {"method": "card", "amount": 5999, "status": "captured"},
    "1044": {"method": "COD", "amount": 1598, "status": "due_on_delivery"},
}

# Seconds each lookup sleeps, to stand in for a network call. 0 in tests that
# don't care; scripts/wismo.py sets it so the parallelism is visible.
LATENCY = 0.0


# Every key the parent and subgraph share needs the same reducer in both.
# operator.or_ on two dicts is `a | b`: a merged copy, right side wins. Keyed
# by order_id, so parallel writers for different orders never collide, and
# re-looking-up an order replaces its old entry.
Contexts = Annotated[dict[str, dict], operator.or_]


class ContextInput(TypedDict):
    order_id: str


class ContextOutput(TypedDict):
    contexts: Contexts


class ContextState(ContextInput, ContextOutput, total=False):
    # Internal. Each lookup writes its own key, so the three parallel writes
    # land in different channels and need no reducer. Two parallel nodes
    # writing the SAME reducer-less key in one step is an error
    # (InvalidUpdateError): LangGraph won't pick a winner for you.
    order: dict
    shipment: dict
    payment: dict


def _fetch(table: dict, order_id: str) -> dict:
    time.sleep(LATENCY)
    return table.get(order_id) or {"found": False}


def fetch_order(state: ContextState) -> dict:
    return {"order": _fetch(ORDERS, state["order_id"])}


def fetch_shipment(state: ContextState) -> dict:
    return {"shipment": _fetch(SHIPMENTS, state["order_id"])}


def fetch_payment(state: ContextState) -> dict:
    return {"payment": _fetch(PAYMENTS, state["order_id"])}


def merge(state: ContextState) -> dict:
    # The "reduce" half. Shapes three partial views into one record, keyed by
    # order so the parent can merge many of these with a plain dict union.
    oid = state["order_id"]
    return {
        "contexts": {
            oid: {
                "order_id": oid,
                "order": state["order"],
                "shipment": state["shipment"],
                "payment": state["payment"],
            }
        }
    }


def build_context_subgraph():
    g = StateGraph(ContextState, input_schema=ContextInput, output_schema=ContextOutput)
    g.add_node("order", fetch_order)
    g.add_node("shipment", fetch_shipment)
    g.add_node("payment", fetch_payment)
    g.add_node("merge", merge)
    for lookup in ("order", "shipment", "payment"):
        g.add_edge(START, lookup)
    # A list of sources is an explicit join: merge runs once, after all three.
    # (Three separate edges would also work here, because all three finish in
    # the same step. The list form still waits if one branch were longer.)
    g.add_edge(["order", "shipment", "payment"], "merge")
    g.add_edge("merge", END)
    # No checkpointer of its own. Used as a node, it inherits the parent's.
    return g.compile()
