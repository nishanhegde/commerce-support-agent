"""Day 45 tests: the context subgraph and the map-reduce WISMO graph. No API calls."""

import json
import time

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from support_agent import context
from support_agent.context import build_context_subgraph
from support_agent.persistence import thread, turn_input
from support_agent.wismo import build_wismo
from test_react import ScriptedModel


class RecordingModel(ScriptedModel):
    """A ScriptedModel that also keeps every prompt it was sent."""

    # A pydantic model, so an extra attribute has to be a declared field.
    # default_factory, not `= []`: the mutable-default trap again.
    prompts: list = Field(default_factory=list)

    def invoke(self, input, *args, **kwargs):
        self.prompts.append(input)
        return super().invoke(input, *args, **kwargs)


def recording(*texts: str) -> RecordingModel:
    return RecordingModel(messages=iter(AIMessage(t) for t in texts))


def records_sent(model: RecordingModel, call: int = -1) -> list[dict]:
    """The order records embedded in the system prompt of one model call."""
    system = model.prompts[call][0].content
    return json.loads(system.split("Order records:\n", 1)[1])


# --- the subgraph on its own -----------------------------------------------


def test_subgraph_returns_only_its_output_schema():
    out = build_context_subgraph().invoke({"order_id": "1042"})

    # order/shipment/payment were internal; only `contexts` comes out.
    assert list(out) == ["contexts"]
    ctx = out["contexts"]["1042"]
    assert ctx["order"]["status"] == "in_transit"
    assert ctx["shipment"]["last_scan"] == "Arrived at Bengaluru hub"
    assert ctx["payment"]["status"] == "captured"


def test_missing_records_come_back_as_not_found():
    ctx = build_context_subgraph().invoke({"order_id": "1044"})["contexts"]["1044"]
    assert ctx["shipment"] == {"found": False}  # pending: nothing shipped yet
    assert ctx["payment"]["status"] == "due_on_delivery"


def test_the_three_lookups_run_in_parallel(monkeypatch):
    monkeypatch.setattr(context, "LATENCY", 0.2)
    start = time.perf_counter()
    build_context_subgraph().invoke({"order_id": "1042"})
    # Sequential would be 0.6s. Generous margin so a slow CI box doesn't flake.
    assert time.perf_counter() - start < 0.45


# --- map-reduce over orders ------------------------------------------------


def test_one_subgraph_run_per_order_then_one_answer():
    model = recording("Both are on their way.")
    graph = build_wismo(model)

    merges = [
        ns
        for ns, update in graph.stream(
            turn_input("where are 1042 and 1043?"), stream_mode="updates", subgraphs=True
        )
        if "merge" in update
    ]

    assert len(merges) == 2  # two Sends, two subgraph runs
    assert len(set(merges)) == 2  # each in its own namespace
    assert len(model.prompts) == 1  # but a single model call
    assert [r["order_id"] for r in records_sent(model)] == ["1042", "1043"]


def test_orders_are_looked_up_in_parallel_too(monkeypatch):
    monkeypatch.setattr(context, "LATENCY", 0.2)
    graph = build_wismo(recording("ok"))
    start = time.perf_counter()
    graph.invoke(turn_input("1042, 1043 and 1044?"))
    # 3 orders x 3 lookups sequentially would be 1.8s.
    assert time.perf_counter() - start < 0.45


def test_repeated_order_ids_are_looked_up_once():
    model = recording("ok")
    out = build_wismo(model).invoke(turn_input("is 1042 late? 1042 was due today"))
    assert out["order_ids"] == ["1042"]
    assert len(records_sent(model)) == 1


def test_unknown_order_is_passed_through_as_not_found():
    model = recording("I couldn't find 9999.")
    build_wismo(model).invoke(turn_input("where is 9999?"))
    [record] = records_sent(model)
    assert record["order"] == {"found": False}


def test_no_order_number_skips_the_lookups():
    model = recording("Which order number?")
    out = build_wismo(model).invoke(turn_input("where is my order?"))

    assert not out.get("contexts")  # the subgraph never ran
    assert "(none)" in model.prompts[0][0].content
    assert out["messages"][-1].content == "Which order number?"


def test_later_turns_only_send_that_turns_orders():
    # With a checkpointer, `contexts` keeps 1042 from turn 1. Turn 2 asks
    # about 1043 only, so only 1043 should reach the model.
    model = recording("first", "second")
    graph = build_wismo(model, checkpointer=InMemorySaver())
    graph.invoke(turn_input("where is 1042?"), thread("t"))
    out = graph.invoke(turn_input("and 1043?"), thread("t"))

    assert sorted(out["contexts"]) == ["1042", "1043"]
    assert [r["order_id"] for r in records_sent(model)] == ["1043"]
