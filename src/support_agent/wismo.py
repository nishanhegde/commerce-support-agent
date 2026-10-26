"""Day 45: map-reduce. "Where are my orders?" across several orders at once.

    START -> extract --Send per order--> context (subgraph) --> answer -> END
                    \\-- no order ids ---------------------------> answer

Map: `fan_out` returns one Send("context", {"order_id": ...}) per order the
customer mentioned. Each Send is a separate run of the context subgraph with
its own input, all in the same superstep, so two orders means two subgraph
runs side by side, each doing its own three lookups in parallel.

Reduce: every run returns {"contexts": {order_id: {...}}}, and the parent's
`contexts` reducer unions them. Then `answer` makes ONE model call with all
of it, instead of the ReAct agent's one round trip per tool call.

Compare with the ReAct graph: there the model decides what to look up, one
turn at a time. Here the graph decides, up front, and the model only writes.
Cheaper and faster for a question whose lookups are known in advance (WISMO
nearly always is); worse when they aren't. That's the "workflow vs agent"
split from Day 11's reading.

Static vs dynamic fan-out, side by side:
  - context.py: three edges from START. The branches are fixed when you
    build the graph.
  - here: Send. The number of branches is decided at runtime, from data.
"""

import json
import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Send

from support_agent.context import Contexts, build_context_subgraph
from support_agent.react import default_model, message_text

SYSTEM = """You are a commerce support agent answering "where is my order"
questions. Answer only from the order records below. They were looked up
just now, so do not say you are checking. Cover every order the customer
asked about, in one reply. If a record says found: false, say you couldn't
find that order. Be brief and concrete.

Order records:
{records}"""

# Order numbers in this shop are 4 digits. A regex rather than an LLM call:
# it's deterministic, free, and good enough to show the graph shape. A real
# extractor would be a small model with structured output (Day 11's routing).
_ORDER_ID = re.compile(r"\b\d{4}\b")


class WismoState(MessagesState):
    order_ids: list[str]
    # Same Annotated type as the subgraph's output, so the reducer matches.
    contexts: Contexts


def build_wismo(
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    llm = model or default_model()

    def extract(state: WismoState) -> dict:
        text = message_text(state["messages"][-1])
        # dict.fromkeys de-duplicates while keeping first-seen order. A set
        # would lose the order the customer mentioned them in.
        return {"order_ids": list(dict.fromkeys(_ORDER_ID.findall(text)))}

    def fan_out(state: WismoState) -> list[Send] | Literal["answer"]:
        if not state["order_ids"]:
            return "answer"
        return [Send("context", {"order_id": oid}) for oid in state["order_ids"]]

    def answer(state: WismoState) -> dict:
        # Only this turn's orders. With a checkpointer, `contexts` keeps
        # entries from earlier turns (the reducer only ever adds).
        contexts = state.get("contexts", {})
        records = [contexts[oid] for oid in state["order_ids"] if oid in contexts]
        system = SYSTEM.format(records=json.dumps(records, indent=2) if records else "(none)")
        return {"messages": [llm.invoke([SystemMessage(system), *state["messages"]])]}

    g = StateGraph(WismoState)
    g.add_node("extract", extract)
    # A compiled graph is a node like any function. LangGraph maps the
    # parent's state into ContextInput and ContextOutput back out.
    g.add_node("context", build_context_subgraph())
    g.add_node("answer", answer)
    g.add_edge(START, "extract")
    # The path list tells the drawing where Sends can go; Send targets aren't
    # visible from the return type the way Literal ones are.
    g.add_conditional_edges("extract", fan_out, ["context", "answer"])
    # Every Send's run must finish before `answer` runs: they're all in one
    # superstep, and `answer` is in the next.
    g.add_edge("context", "answer")
    g.add_edge("answer", END)
    return g.compile(checkpointer=checkpointer)
