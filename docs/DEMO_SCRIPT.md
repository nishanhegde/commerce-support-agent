# Demo: the angry customer, end to end (about 4 minutes)

Record the terminal. Three windows' worth of story, one terminal.

## Setup (off camera)

```bash
cd commerce-support-agent
rm -f data/audit.sqlite data/checkpoints.sqlite      # clean slate (literal paths, in this repo)
```

## 1. The conversation (2 min)

```bash
uv run python scripts/chat.py --supervisor --customer C-100 --thread demo
```

Type, in order:

1. `This is ridiculous. My Trail Runner 4 was supposed to arrive over a week ago and it STILL isn't here. I don't want it anymore, I want my money back.`
2. `Yes. Refund it.` — the graph stops at the approval prompt. **Pause here and say why:** ₹7,999 is over the ₹5,000 rule, so a person decides, and the rule, not the model, made that call.
3. `y` to approve (name yourself as reviewer).
4. Optional: `Actually, get me a manager.` to show the escalation and the bot stepping out.

Quit with an empty line. The **trace summary** prints: model calls, tokens, cost, latency per node.

## 2. Show your working (1 min)

```bash
uv run python scripts/audit.py --thread demo
cat data/handoffs/HO-0001.md      # only if you escalated
```

Point at: the supervisor's routing, `returns_agent/tools`, the APPROVAL row (who, which rule, why), and in the handoff document the line between the AI-generated summary and the record copied from the audit log.

## 3. What happens when it breaks (1 min)

```bash
uv run pytest tests/test_chaos.py -v
```

Say: the real MCP server gets `SIGKILL`ed mid-conversation; reads are retried, writes never are, the customer gets an honest answer, and the thread keeps working.

## Say out loud

- A single agent is the better design for this problem (about half the model calls); the supervisor is built to find that out, not because it's needed. The README has the numbers and the three conditions under which multi-agent wins.
- The approval gate is policy code, not a prompt.
- Not run live: Days 51-57 (API credit). Offline tests cover them; say so if asked.
