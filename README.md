# commerce-support-agent

Month 3: LangGraph customer-support agent over the `mcp-commerce` tool layer.

## Single agent, supervisor, handoffs, shared state: when is multi-agent worth it?

The repo has all four designs, built over the same tools, policy and audit
log:

| Design | Code | Who decides the next step | What each agent sees |
|---|---|---|---|
| Single agent, all tools | `react.build_explicit` | one ReAct loop | everything |
| Supervisor + isolated workers | `supervisor.build_supervisor()` | supervisor, after every report | a worker sees only its brief |
| ... + handoffs | the same, `transfer_to_*` tools | a worker can pass the case on directly | the receiver sees the sender's note |
| ... + shared state | `build_supervisor(shared=True)` | supervisor | conversation + shared case file |

### What each costs, on the Day 50 scenario

The angry customer, late order 1048, refund over the approval threshold, two
customer turns. Model calls per design:

| Design | Model calls | How the number was obtained |
|---|---|---|
| Single agent | **5** | measured: Day 50 live run 2 (`failures.md`) |
| Supervisor | ~11 | counted from the graph: 5 supervisor calls (route, route, finish; route, finish) + 6 worker calls |
| Supervisor + handoff | ~10 | the handoff saves the supervisor call between order_agent and returns_agent |
| Shared state | ~11, with longer prompts | as the supervisor, and every worker prompt also carries the conversation and case file |

The multi-agent numbers are worked out, not measured: the API credit ran
out before they could be run live. The ratio is the point, and it follows
from the structure: every hop through a supervisor is an extra model call,
and every worker re-establishes context the single agent already had. The
same goes for latency, since the calls are sequential.

### What multi-agent buys you, and whether it applies here

| Claimed benefit | Real? | In this repo |
|---|---|---|
| Smaller prompts and tool lists per agent | Yes, once a single agent has too many tools to choose from reliably | **No.** The single agent has 7 tools; that's well within what one prompt handles |
| Hard permission boundaries | Weaker than it looks | **No.** `policy.py` gates actions per call, inside every design. Moving a tool to another agent isn't a security boundary; the policy is |
| A different model per role (cheap router, strong worker) | Yes | **Possible**, not done: routing is the obvious Haiku candidate, but not measured |
| Parallel work | Yes, for independent lookups | **Not used.** Day 45's map-reduce got parallelism inside one graph without extra agents |
| Separate ownership: teams, prompts, evals per domain | Yes, organisationally | **No.** One developer, one domain |

### The failure modes it adds

- **Isolated workers lose context.** Anything the supervisor leaves out of a
  brief, the worker doesn't know. "The customer has already confirmed" is
  the classic miss, and it costs a whole extra turn.
- **Handoffs lose context in the note,** for the same reason, and can
  bounce back and forth. Hence `max_handoffs`, and handoffs counting as
  delegations.
- **Shared state brings back the single agent's context size,** spread over
  more calls, and adds write conflicts. Those need rules: `case.py` uses
  field ownership, history on overwrite, append-only notes, and a reducer
  so the rules hold for parallel writers.
- **Debugging is harder.** A wrong answer can come from the routing, a
  brief, a worker, or the summary. The audit log names each worker
  (`returns_agent/tools`) and records every routing decision and handoff;
  without that, multi-agent runs are close to impossible to debug.

### The honest answer

For this agent, a single ReAct loop with the policy layer is the better
design: about half the model calls, one prompt to debug, and nothing the
multi-agent versions do that it can't. Multi-agent is worth it when at
least one of these holds:

1. **Too many tools for one agent.** Tool choice degrades long before the
   context window fills. If evals show wrong-tool calls rising as tools are
   added, split by domain.
2. **The work really is parallel and independent,** like research across
   sources. Then the extra calls run at the same time instead of in a row.
3. **The roles need different models or different owners.** A cheap
   router in front of expensive specialists, or one team per domain
   shipping its own prompts and evals.

The cheapest step that's still worth having isn't multi-agent at all: a
**router** (one cheap classification call, or Day 41's keyword
`classify`) in front of narrower single agents. If the traces show one hop
per turn, that's what the supervisor was anyway.

## Escalation, and why it looks like contact-centre integration

`escalation.py` hands a case to a person in three steps: **summarise** the
conversation (one model call), **package** a handoff document, and
**transfer** it: queue it, write it out as `data/handoffs/<ticket>.md` and
`.json`, and take the bot out of the thread. That's the shape of a
bot-to-agent transfer in a contact-centre platform, and building it by hand
makes the reasons for that shape obvious:

- **The document is an integration contract.** The `.json` is what a
  real transfer API would receive: a ticket or case with routing
  attributes (queue, priority), the customer and order ids, and the
  transcript attached. The `.md` is the agent's screen pop. In a commerce
  stack the ticket lives in the service system, not the shop, and the
  `customer_id` and order numbers are the keys that link it back to the
  commerce data. Without those keys, the human's screen pop can't show the
  order.
- **Summary and record are kept apart on purpose.** Agents on the receiving
  end stop trusting AI summaries the first time one is wrong, and then they
  re-ask the customer, which is the thing a handover exists to prevent. So
  the summary is labelled as generated, and the facts under it (approvals
  with reviewer and rule, handoffs, the transcript) are copied from the
  audit log verbatim. Building it showed why: the first version reported a
  *rejected* refund as `create_return ran (error)`, which a person would
  read as a system failure.
- **One owner per conversation.** Once transferred, the bot stops answering:
  new messages go onto the ticket (`with_human`) without a model call. A
  contact centre does the same, for the same reason: two parties answering
  one customer, neither aware of what the other said.
- **This is a cold transfer:** the case goes to a queue, and whoever picks
  it up reads the document. A warm transfer, where the bot introduces the
  human in the live chat, needs agent-availability data this stub doesn't
  have. That's also why the reply promises no channel and no time.
- **Missing before this could face customers:** checking agent availability
  and business hours before transferring, carrying over the customer's
  verified identity, redacting personal and payment data from the
  transcript, a path back from human to bot, and closing the ticket.
