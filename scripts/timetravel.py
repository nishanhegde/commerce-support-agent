"""Inspect, replay, and fork a thread's checkpoints.

    uv run python scripts/timetravel.py --thread alice                 # list checkpoints
    uv run python scripts/timetravel.py --thread alice --replay ab12cd34
    uv run python scripts/timetravel.py --thread alice --fork ab12cd34 \\
        --tool-result '{"order_id": "1042", "status": "delivered"}'

The id to pass is the 8-character first column of the listing: the *tail* of
the checkpoint id, because ids are time-ordered and their heads barely differ. --fork
needs a checkpoint whose next node is `agent` and whose last message is a
tool result.
"""

import argparse

from dotenv import load_dotenv

from support_agent.persistence import (
    describe,
    find,
    fork_tool_result,
    history,
    replay,
    sqlite_checkpointer,
)
from support_agent.react import build_explicit, final_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="demo")
    parser.add_argument("--replay", metavar="CHECKPOINT")
    parser.add_argument("--fork", metavar="CHECKPOINT")
    parser.add_argument("--tool-result", help="replacement tool output (JSON) for --fork")
    args = parser.parse_args()
    load_dotenv()

    with sqlite_checkpointer() as saver:
        graph = build_explicit(checkpointer=saver)

        if args.replay:
            out = replay(graph, find(graph, args.thread, args.replay))
            print(f"replayed -> {final_text(out)}\n")
        elif args.fork:
            if not args.tool_result:
                parser.error("--fork needs --tool-result")
            out = fork_tool_result(graph, find(graph, args.thread, args.fork), args.tool_result)
            print(f"forked -> {final_text(out)}\n")

        # Always end on the listing, so you see where a replay/fork landed.
        # It shows every branch; follow the `<- parent` column to see the tree.
        for snap in history(graph, args.thread):
            print(describe(snap))


if __name__ == "__main__":
    main()
