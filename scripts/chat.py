"""Multi-turn chat that survives restarts. Quit, run again with the same
--thread, and the agent still remembers the conversation.

Day 44: every step streams as it happens, and a cancellation stops for your
approval. Quit at the approval prompt (Ctrl-D) and the thread stays paused;
run again with the same --thread and you're asked again.

Run: uv run python scripts/chat.py --thread alice
Try: "please cancel order 1044, I ordered the wrong size"
"""

import argparse
import json

from dotenv import load_dotenv

from support_agent.hitl import decision, pending_approval, render
from support_agent.persistence import DB_PATH, sqlite_checkpointer, thread, turn_input
from support_agent.react import build_explicit


def review(pending: list[dict]):
    """Ask the human at the terminal. Raises EOFError on Ctrl-D, which leaves
    the thread paused."""
    print("  ** approval needed **")
    for c in pending:
        print(f"     {c['name']}({json.dumps(c['args'])})")
    if input("approve? [y/N] ").strip().lower() == "y":
        return decision(True)
    return decision(False, input("reason> ").strip())


def run(graph, graph_input, config: dict) -> None:
    """Stream one turn, stopping for approval as often as the agent asks."""
    while True:
        for update in graph.stream(graph_input, config, stream_mode="updates"):
            for node, delta in update.items():
                for line in render(node, delta):
                    print(line)
        pending = pending_approval(graph, config)
        if not pending:
            return
        # Resuming is just another stream() call, with a Command as input.
        graph_input = review(pending)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="demo")
    args = parser.parse_args()
    load_dotenv()

    with sqlite_checkpointer() as saver:
        graph = build_explicit(checkpointer=saver)
        config = thread(args.thread)
        prior = graph.get_state(config).values.get("messages", [])
        print(f"thread={args.thread!r}  db={DB_PATH}")
        print(f"resuming with {len(prior)} messages" if prior else "new conversation")
        print("(empty line or Ctrl-D to quit)\n")

        try:
            # A thread left paused by a previous run has to be settled first:
            # a new user message can't go in while a tool_use is unanswered.
            if pending := pending_approval(graph, config):
                print("this thread was paused waiting for approval")
                run(graph, review(pending), config)
                print()

            while True:
                text = input("you> ").strip()
                if not text:
                    break
                run(graph, turn_input(text), config)
                print()
        except EOFError:
            pass


if __name__ == "__main__":
    main()
