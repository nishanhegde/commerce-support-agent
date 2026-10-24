"""Print the graph as Mermaid and write docs/graph.mmd (paste into mermaid.live)."""

from pathlib import Path

from support_agent.graph import build_graph

mermaid = build_graph().get_graph().draw_mermaid()
out = Path(__file__).resolve().parent.parent / "docs" / "graph.mmd"
out.parent.mkdir(exist_ok=True)
out.write_text(mermaid)
print(mermaid)
