"""Multi-turn chat that survives restarts. Quit, run again with the same
--thread, and the agent still remembers the conversation.

Day 44: every step streams as it happens, and a cancellation stops for your
approval. Quit at the approval prompt (Ctrl-D) and the thread stays paused;
run again with the same --thread and you're asked again.

Day 46: --mcp swaps the in-memory tools for mcp-commerce's, started as a
stdio subprocess against its OCC mock. The graph, checkpoints and approval
gate are the same; only the tools (and so the gated set) change.

Run: uv run python scripts/chat.py --thread alice
Try: "please cancel order 1044, I ordered the wrong size"
Try: "I want to send back everything from order 1046" (Day 47: returns)

Run: uv run python scripts/chat.py --mcp --thread wismo
Try: "I ordered running shoes in August and they still haven't arrived"
"""

import argparse
import json
from contextlib import ExitStack
from datetime import date

from dotenv import load_dotenv

from support_agent.hitl import decision, pending_approval, render
from support_agent.mcp_bridge import MCPBridge, mcp_commerce_server
from support_agent.persistence import DB_PATH, sqlite_checkpointer, thread, turn_input
from support_agent.react import build_explicit

# Month 2's demo prompt (mcp-commerce/examples/demo.py), trimmed. The mock's
# orders are dated 2026, so "today" matters: it's how the model can tell a
# shipment is late.
MCP_SYSTEM = """You are the customer support agent for an online sports retailer.
Today is {today}.

Use your tools for every fact: order status, tracking, stock, prices. Never
guess a carrier, a date, or a stock level. If the customer has no order
number, find it with get_order_history.

A shipment is delayed when its estimated_delivery is before today and it has
not been delivered. When an order is delayed, say so plainly first (carrier,
tracking id, original estimate), then check whether the items are in stock
now and offer concrete options. Don't create carts until the customer picks
an option. A human approves each cart change before it runs, so a result
you see is final. Only offer what your tools can do, and never promise that
someone else will follow up.

Be brief: a few short paragraphs, prices in the order's currency."""


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
    parser.add_argument("--mcp", action="store_true", help="use mcp-commerce's tools")
    args = parser.parse_args()
    load_dotenv()

    # ExitStack: enter a variable set of context managers, all closed in
    # reverse order on the way out. The bridge is only one of them with --mcp.
    with ExitStack() as stack:
        saver = stack.enter_context(sqlite_checkpointer())
        if args.mcp:
            bridge = stack.enter_context(MCPBridge(mcp_commerce_server()))
            graph = build_explicit(
                checkpointer=saver,
                # Month 2's demo allowed 10 rounds. A WISMO chain is ~4, and
                # recovering from a server error adds a few more.
                max_iterations=10,
                tools=bridge.tools,
                needs_approval=bridge.needs_approval,
                system=MCP_SYSTEM.format(today=date.today().isoformat()),
            )
            print(f"mcp-commerce: {', '.join(t.name for t in bridge.tools)}")
            print(f"needs approval: {', '.join(sorted(bridge.needs_approval))}")
        else:
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
