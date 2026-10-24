"""Run the prebuilt and explicit ReAct agents side by side against real Claude.

Needs ANTHROPIC_API_KEY in this repo's .env (gitignored).
Run: uv run python scripts/react_demo.py
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

from support_agent.react import build_explicit, build_prebuilt, final_text

QUESTIONS = [
    "Where is my order 1042?",
    # Two independent lookups: a good model batches them into one turn.
    "Has order 1042 shipped, and do you have insoles in stock?",
    # Sequential: can't search for an alternative until it knows what's in the order.
    "Can you find me something cheaper than what's in order 1042?",
]


def trace(result: dict) -> None:
    for m in result["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            calls = ", ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
            print(f"    -> {calls}")
        elif isinstance(m, ToolMessage):
            print(f"    <- [{m.status}] {m.content[:100]}")


def main() -> None:
    load_dotenv()
    agents = {"prebuilt": build_prebuilt(), "explicit": build_explicit()}

    for name, graph in agents.items():
        print(f"\n##### {name}\n{graph.get_graph().draw_mermaid()}")

    for q in QUESTIONS:
        print(f"\n=== {q}")
        for name, graph in agents.items():
            out = graph.invoke({"messages": [("user", q)]})
            print(f"  [{name}]")
            trace(out)
            print(f"    {final_text(out)}")

    # The guard firing for real: one iteration means the model asks for a tool
    # and never gets to see the result.
    print("\n=== guard: max_iterations=1")
    out = build_explicit(max_iterations=1).invoke({"messages": [("user", QUESTIONS[0])]})
    print(f"    {final_text(out)}")


if __name__ == "__main__":
    main()
