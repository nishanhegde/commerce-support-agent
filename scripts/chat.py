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

Day 48: --customer picks who is signed in (C-100 or C-200). Their profile
loads at the start of each new thread, and notes saved in one thread are
there in the next. Same --thread: the conversation is remembered. New
--thread, same --customer: only the customer is.

Run: uv run python scripts/chat.py --customer C-100 --thread a1
Try: "I'm a UK 9 in road shoes, by the way" then quit, and
     uv run python scripts/chat.py --customer C-100 --thread a2
Try: "what size should I order the Road Runner 2 in?"

Day 49: refunds over ₹5,000 stop for approval and smaller ones don't; every
decision and every node run is written to data/audit.sqlite. Read it back
with scripts/audit.py --thread <name>.

Try: "return one pair of the Road Runner 2 from 1046"   (₹4,499: no prompt)
Try: "return the jacket from 1043"                      (₹5,999: approval)

Run: uv run python scripts/chat.py --mcp --thread wismo
Try: "I ordered running shoes in August and they still haven't arrived"
"""

import argparse
import getpass
import json
from contextlib import ExitStack

from dotenv import load_dotenv

from support_agent.audit import AUDIT_PATH, AuditLog
from support_agent.hitl import decision, pending_approval, render
from support_agent.mcp_bridge import MCPBridge, mcp_commerce_server
from support_agent.policy import by_name
from support_agent.memory import CustomerContext, seed_profiles
from support_agent.persistence import (
    DB_PATH,
    STORE_PATH,
    sqlite_checkpointer,
    sqlite_store,
    thread,
    turn_input,
)
from support_agent.react import build_explicit

# Month 2's demo prompt (mcp-commerce/examples/demo.py), trimmed. Day 50:
# the graph now appends today's date to every system prompt, so it's no
# longer formatted in here.
MCP_SYSTEM = """You are the customer support agent for an online sports retailer.

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
        print(f"     why: {c['why']}")
    # Day 49: the OS login stands in for the reviewer's identity. It goes in
    # the approval log next to the decision.
    reviewer = getpass.getuser()
    if input("approve? [y/N] ").strip().lower() == "y":
        return decision(True, reviewer=reviewer)
    return decision(False, input("reason> ").strip(), reviewer=reviewer)


def run(graph, graph_input, config: dict, context: CustomerContext | None) -> None:
    """Stream one turn, stopping for approval as often as the agent asks.

    `context` goes into every stream() call, resumes too: it isn't saved in
    the checkpoint, so a resume without it would run with no customer.
    """
    while True:
        for update in graph.stream(graph_input, config, stream_mode="updates", context=context):
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
    parser.add_argument("--customer", default="C-100", help="signed-in customer (not with --mcp)")
    args = parser.parse_args()
    load_dotenv()

    # ExitStack: enter a variable set of context managers, all closed in
    # reverse order on the way out. The bridge is only one of them with --mcp.
    context = None
    with ExitStack() as stack:
        saver = stack.enter_context(sqlite_checkpointer())
        audit = stack.enter_context(AuditLog(AUDIT_PATH))
        if args.mcp:
            bridge = stack.enter_context(MCPBridge(mcp_commerce_server()))
            graph = build_explicit(
                checkpointer=saver,
                # Month 2's demo allowed 10 rounds. A WISMO chain is ~4, and
                # recovering from a server error adds a few more.
                max_iterations=10,
                tools=bridge.tools,
                policy=by_name(bridge.needs_approval),
                audit=audit,
                system=MCP_SYSTEM,
            )
            print(f"mcp-commerce: {', '.join(t.name for t in bridge.tools)}")
            print(f"needs approval: {', '.join(sorted(bridge.needs_approval))}")
        else:
            # Day 48: long-term memory. Not with --mcp, whose orders are a
            # different data set; there the customer is the OCC user.
            store = stack.enter_context(sqlite_store())
            seed_profiles(store)
            graph = build_explicit(checkpointer=saver, store=store, audit=audit)
            context = CustomerContext(args.customer)
            print(f"customer={args.customer!r}  memory={STORE_PATH}")
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
                run(graph, review(pending), config, context)
                print()

            while True:
                text = input("you> ").strip()
                if not text:
                    break
                run(graph, turn_input(text), config, context)
                print()
        except EOFError:
            pass


if __name__ == "__main__":
    main()
