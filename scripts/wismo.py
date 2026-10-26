"""Watch the map-reduce run: every subgraph step, timed, then one answer.

Each lookup sleeps 0.5s to stand in for a network call. Three orders means
nine lookups; run in sequence that's 4.5s. Watch the clock.

Also writes docs/wismo.mmd, the graph with the subgraph drawn inside it.

Needs ANTHROPIC_API_KEY in this repo's .env (gitignored).
Run: uv run python scripts/wismo.py ["your question"]
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from support_agent import context
from support_agent.persistence import turn_input
from support_agent.react import final_text
from support_agent.wismo import build_wismo

QUESTION = "Where are my orders 1042, 1043 and 1044, and have I been charged for them?"


def main() -> None:
    load_dotenv()
    context.LATENCY = 0.5
    question = sys.argv[1] if len(sys.argv) > 1 else QUESTION
    graph = build_wismo()

    # xray=True draws subgraphs expanded instead of as one box.
    out = Path(__file__).resolve().parent.parent / "docs" / "wismo.mmd"
    out.write_text(graph.get_graph(xray=True).draw_mermaid())

    print(f"you> {question}\n")
    start = time.perf_counter()
    # subgraphs=True streams the subgraph's own steps too. The namespace says
    # which run they came from: () is the parent, ("context:<task id>",) is
    # one Send's run of the subgraph.
    for ns, update in graph.stream(
        turn_input(question), stream_mode="updates", subgraphs=True
    ):
        for node, delta in update.items():
            where = ns[0].split(":")[1][:8] if ns else "parent"
            detail = ""
            if node == "extract":
                detail = f"order_ids={delta['order_ids']}"
            elif node == "merge":
                detail = f"order {next(iter(delta['contexts']))}"
            print(f"  {time.perf_counter() - start:5.2f}s  {where:<8} {node:<9} {detail}")
        if not ns and "answer" in update:
            print(f"\nagent> {final_text(update['answer'])}")
    print(f"\ndiagram: {out}")


if __name__ == "__main__":
    main()
