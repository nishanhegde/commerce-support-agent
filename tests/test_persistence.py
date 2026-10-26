"""Day 43 tests: SQLite checkpoints, resume, replay, fork. No API calls."""

import json

from langchain_core.messages import AIMessage, ToolMessage

from support_agent.persistence import (
    ask,
    fork_tool_result,
    history,
    sqlite_checkpointer,
    thread,
)
from support_agent.react import build_explicit, final_text
from test_react import call, scripted


def order_lookup_then(answer: str) -> list[AIMessage]:
    return [
        AIMessage("", tool_calls=[call("get_order_status", {"order_id": "1042"})]),
        AIMessage(answer),
    ]


def test_conversation_survives_a_restart(tmp_path):
    db = tmp_path / "cp.sqlite"

    # "Process 1": one turn, then the saver (and its connection) is closed.
    with sqlite_checkpointer(db) as saver:
        graph = build_explicit(scripted(AIMessage("Hi! How can I help?")), checkpointer=saver)
        ask(graph, "t1", "hello")

    # "Process 2": a new saver and a new graph over the same file.
    with sqlite_checkpointer(db) as saver:
        graph = build_explicit(scripted(AIMessage("Order 1042 is in transit.")), checkpointer=saver)
        assert len(graph.get_state(thread("t1")).values["messages"]) == 2  # loaded, not re-run
        out = ask(graph, "t1", "where is 1042?")

    assert [m.type for m in out["messages"]] == ["human", "ai", "human", "ai"]


def test_threads_are_isolated(tmp_path):
    with sqlite_checkpointer(tmp_path / "cp.sqlite") as saver:
        graph = build_explicit(scripted(AIMessage("a"), AIMessage("b")), checkpointer=saver)
        ask(graph, "alice", "hi")
        out = ask(graph, "bob", "hi")

    assert len(out["messages"]) == 2


def test_iterations_reset_each_turn(tmp_path):
    # Each turn uses 2 iterations. With max_iterations=2 and no reset, turn 2
    # would start at 2 and hit the guard after its first tool call.
    model = scripted(*order_lookup_then("first"), *order_lookup_then("second"))
    with sqlite_checkpointer(tmp_path / "cp.sqlite") as saver:
        graph = build_explicit(model, max_iterations=2, checkpointer=saver)
        ask(graph, "t", "where is 1042?")
        out = ask(graph, "t", "and now?")

    assert final_text(out) == "second"


def test_history_records_every_superstep(tmp_path):
    with sqlite_checkpointer(tmp_path / "cp.sqlite") as saver:
        graph = build_explicit(scripted(*order_lookup_then("done")), checkpointer=saver)
        ask(graph, "t", "where is 1042?")
        snaps = history(graph, "t")

    # input -> (start) -> agent -> tools -> agent -> end
    assert [s.next for s in snaps] == [("__start__",), ("agent",), ("tools",), ("agent",), ()]


def test_fork_edits_one_tool_result_and_keeps_the_original_branch(tmp_path):
    model = scripted(
        *order_lookup_then("It's on the way."),
        AIMessage("It was delivered."),  # what the model says on the forked branch
    )
    with sqlite_checkpointer(tmp_path / "cp.sqlite") as saver:
        graph = build_explicit(model, checkpointer=saver)
        ask(graph, "t", "where is 1042?")
        original = history(graph, "t")
        after_tools = next(s for s in original if s.next == ("agent",) and s.metadata["step"] > 1)

        forked = fork_tool_result(graph, after_tools, json.dumps({"status": "delivered"}))

        # The fork replaced the tool result rather than adding a second one.
        [tm] = [m for m in forked["messages"] if isinstance(m, ToolMessage)]
        assert json.loads(tm.content) == {"status": "delivered"}
        assert final_text(forked) == "It was delivered."

        # The original run's final checkpoint is still there, untouched.
        assert final_text(graph.get_state(original[-1].config).values) == "It's on the way."
