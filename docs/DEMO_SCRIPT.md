# Demo: the angry customer, end to end

`docs/demo.cast` (and `demo.gif`) is a real run of this flow against Claude,
single agent, recorded 2026-10-28. Nothing in it is scripted except the
customer's typing and the reviewer's `y`: every reply is the model's.

## What it shows

1. **The conversation.** `scripts/chat.py --customer C-100`: an angry customer
   with a late order and no order number. The agent finds order 1048,
   checks eligibility, and asks before acting. After "Yes. Refund it." the
   graph stops: ₹7,999 is over the ₹5,000 rule, so a person decides, and the
   rule, not the model, made that call. The run ends with the trace summary
   (calls, tokens, cost, latency).
2. **The audit log.** `scripts/audit.py --thread <name>`: every node, the
   pause, and the approval row (who, which rule, why).
3. **When it breaks.** `uv run pytest tests/test_chaos.py -v`: the real MCP
   server is `SIGKILL`ed mid-conversation. Reads are retried, writes never
   are, and the thread keeps working.

## Re-recording it

The recording is driven by an `expect` script that types the customer's
lines and answers the approval prompt, wrapped in `asciinema`:

```bash
asciinema rec --headless --overwrite --window-size 100x38 -i 2 \
  -c "expect demo.exp" docs/demo.cast
agg --speed 1.4 --font-size 14 docs/demo.cast docs/demo.gif
```

`demo.exp` is not in the repo. It spawns `chat.py` and waits for `you> `
before each line; for the approval prompt use `expect -ex "approve? \[y/N\] "`
(an unescaped `[y/N]` is a glob, and waits out the whole timeout).
Needs `ANTHROPIC_API_KEY` in `.env`; one take costs about $0.12.

## Worth saying out loud

- A single agent is the better design for this problem (about half the
  model calls). The supervisor exists to find that out; the README has the
  numbers and the three conditions under which multi-agent wins.
- The approval gate is policy code, not a prompt.
- "Refuse delivery" in the last reply is not invented: it comes from
  `create_return`'s own `next_step`. The supervisor path, by contrast, did
  offer options nothing grounded (`failures.md` F8).
- The supervisor flow has run live only twice, and found three problems.
