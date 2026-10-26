"""Day 43: checkpoints. Resume a conversation across restarts, and time-travel.

A checkpointer saves the full graph state after every superstep, keyed by
(thread_id, checkpoint_id). Two things fall out of that:

1. Resume. Month 1's CLI kept `messages` in a Python list and wrote it to
   conversations/*.json by hand. Here you pass only the *new* user message
   plus a thread_id; LangGraph loads the thread's latest checkpoint, and the
   add_messages reducer appends to the history it finds there.

2. Time travel. Every intermediate state is kept, not just the latest. You
   can list them, re-run from any of them (replay), or edit one and run from
   the edit (fork). For an agent, that turns "why did it answer that?" from
   rerunning the whole conversation and hoping into a controlled experiment:
   change exactly one tool result and see if the answer changes.

Checkpoints are a parent-linked chain, not a list that gets overwritten. A fork
adds a new branch off an old checkpoint; the original branch is still there.
Think git: checkpoint_id is a commit, the thread's latest checkpoint is HEAD.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import StateSnapshot

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "checkpoints.sqlite"


@contextmanager
def sqlite_checkpointer(path: Path = DB_PATH) -> Iterator[SqliteSaver]:
    """A SqliteSaver over a file, closed when the `with` block ends.

    @contextmanager turns a generator into a with-able object: code before
    `yield` is setup, code after is teardown. Java's try-with-resources, without
    writing a class that implements AutoCloseable.
    """
    path.parent.mkdir(exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


def thread(thread_id: str) -> dict:
    """The config that selects a conversation. Every call on a thread needs it."""
    return {"configurable": {"thread_id": thread_id}}


def turn_input(text: str) -> dict:
    # Only the new message goes in; the rest comes from the checkpoint.
    #
    # `iterations: 0` matters. The checkpoint holds ALL state, including the
    # counter, so without resetting it a long conversation would carry turn
    # 1's iterations into turn 5 and hit max_iterations for no reason. Month 1
    # got this for free because Run() was a fresh object per call. Here, any
    # per-turn value has to be reset by the caller on purpose.
    return {"messages": [("user", text)], "iterations": 0}


def ask(graph, thread_id: str, text: str) -> dict:
    return graph.invoke(turn_input(text), thread(thread_id))


# ---------------------------------------------------------------------------
# Time travel
# ---------------------------------------------------------------------------


def history(graph, thread_id: str) -> list[StateSnapshot]:
    """Every checkpoint on the thread, all branches, oldest first.

    get_state_history returns newest first and does NOT follow one branch:
    after a fork, the abandoned original checkpoints are still listed, in
    time order. The tree is only visible through each snapshot's
    parent_config, which `describe` prints. Reversed here because a run reads
    more naturally top to bottom.
    """
    return list(reversed(list(graph.get_state_history(thread(thread_id)))))


def describe(snap: StateSnapshot) -> str:
    """One line per checkpoint: id, parent, step, what wrote it, what runs next,
    last message. Two lines with the same parent are a fork point."""
    step = snap.metadata.get("step")
    source = snap.metadata.get("source")  # "input", "loop", or "update" (a fork)
    nxt = ",".join(snap.next) or "END"
    msgs = snap.values.get("messages", [])
    last = _summarise(msgs[-1]) if msgs else "-"
    cid = snap.config["configurable"]["checkpoint_id"][-8:]
    parent = snap.parent_config["configurable"]["checkpoint_id"][-8:] if snap.parent_config else "-"
    return f"{cid} <- {parent:<8}  step={step:>2}  {source:<6}  next={nxt:<9}  {last}"


def _summarise(m) -> str:
    if isinstance(m, HumanMessage):
        return f"user: {m.content[:60]}"
    if isinstance(m, AIMessage) and m.tool_calls:
        return "ai -> " + ", ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
    if isinstance(m, AIMessage):
        return f"ai: {str(m.content)[:60]}"
    if isinstance(m, ToolMessage):
        return f"tool[{m.status}]: {m.content[:60]}"
    return type(m).__name__


def find(graph, thread_id: str, checkpoint_suffix: str) -> StateSnapshot:
    """Look up a checkpoint by the tail of its id, as `describe` prints it."""
    for snap in graph.get_state_history(thread(thread_id)):
        if snap.config["configurable"]["checkpoint_id"].endswith(checkpoint_suffix):
            return snap
    raise KeyError(f"no checkpoint ending {checkpoint_suffix!r} on thread {thread_id!r}")


def replay(graph, snap: StateSnapshot) -> dict:
    """Re-run from a checkpoint, unchanged.

    Passing None as input with a config that names a checkpoint_id means:
    load that state, run whatever `snap.next` says. Nodes before it are not
    re-executed. Nodes after it ARE, so the model is called again and may
    answer differently. That nondeterminism is itself useful to see.
    """
    return graph.invoke(None, snap.config)


def fork_tool_result(graph, snap: StateSnapshot, new_content: str) -> dict:
    """Replace the latest tool result in `snap`, then run on from the edit.

    `snap` must be a checkpoint taken right after the tools node (next ==
    ("agent",)). The edited ToolMessage keeps the original's id, and
    add_messages treats a matching id as replace, not append, so the history
    shows the new result where the old one was.

    as_node="tools" makes LangGraph treat the edit as if the tools node had
    written it, so routing continues from there (route_after_tools -> agent)
    exactly as it would have in the real run.
    """
    old = next(m for m in reversed(snap.values["messages"]) if isinstance(m, ToolMessage))
    edited = old.model_copy(update={"content": new_content})
    fork_config = graph.update_state(snap.config, {"messages": [edited]}, as_node="tools")
    return graph.invoke(None, fork_config)
