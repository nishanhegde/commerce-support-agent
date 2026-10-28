# End-to-end scenario — Day 50

`scripts/e2e.py`, run against `claude-opus-5` on 2026-10-27, with the date
pinned by the real clock to 2026-10-27. The customer is C-100 (Asha Rao). The
order is 1048 (Trail Runner 4, ₹7,999, BlueDart, estimated 2026-10-18, so 9
days late). The two scripted customer lines are in `src/support_agent/e2e.py`.

Each run is judged in five stages: triage → order agent → eligibility →
approval → resolution. Transcripts are reproducible by re-running the
script; this file is the judged outcome, which is the part a rerun doesn't
keep.

## Runs

| Run | Code | Reviewer | Stages | Notes |
|-----|------|----------|--------|-------|
| 1 | before fixes | approves | **2/5** | F1–F4 found |
| 2 | F1–F4 fixed | approves | **5/5** | F5 found |
| 3 | F1–F4 fixed | rejects | 4/5 | F6 found; the judge was also wrong (expected a return after a rejection) |
| 4 | F6 fixed, judge fixed | rejects | **5/5** | run 2026-10-28 after credit was added: 5 model calls, $0.1156, 21s |

Run 4 (2026-10-28) confirmed the F6 fix live: the reviewer's rejection
was reported to the customer as "not approved", with no return and no
escalation the customer didn't ask for.

## Failures found, and what was done

### F1: no way to refund a late, in-transit order — fixed
**Seen (run 1):** the customer wants their money back for a shipment 9 days
late. `cancel_order` refused it (only pending orders can be cancelled), and
`check_return_eligibility` refused it ("hasn't been delivered yet"). Claude
escalated to a human, because no tool could say yes.
**Cause:** a missing business rule, not a model error. The agent did the
right thing with the tools it had.
**Fix:** `returns.eligibility` now treats an in-transit order more than
`LATE_REFUND_DAYS` (7) past its estimate as refundable without delivery
(`delayed_shipment: true`). The window and category rules don't apply,
since the parcel was never delivered or opened. The quantity rule still
does. `create_return` records it as `carrier_recall`; the customer sends
nothing back and refuses the parcel if it turns up.
**Tests:** `test_f1_*` in `tests/test_e2e.py`.

### F2: a human was asked to approve a cancellation that couldn't happen — fixed
**Seen (run 1):** `cancel_order(1048)` paused for approval, the reviewer
approved it, and then the tool refused it as not pending. That wasted the
reviewer's time and left a misleading "approved" row in the audit log.
**Cause:** the Day 49 policy gated every cancel by name, without the
eligibility pre-check it already did for returns.
**Fix:** `policy.default_policy` returns `cancel_ineligible` (no human) for
any order that isn't pending.
**Tests:** `test_f2_*`.

### F3: promised a follow-up that nothing performs — fixed in the prompt, unverified live
**Seen (run 1):** "They'll follow up by email, which is your contact
preference on file." `escalate_to_human` is a stub and pages nobody. Its
docstring only ruled out promising a response *time*.
**Fix:** the `escalate_to_human` docstring now says to give the ticket id,
say a person will review it, and promise nothing about how or when they'll
be contacted.
**Tests:** none. This is model behaviour, so only a live run shows it. Run 3
said "A person will review it" with no channel promised, which is the
intended wording, but that's one sample.

### F4: no date in the system prompt — fixed
**Seen (run 1):** Claude called 1048 "the overdue one" by comparing the two
estimates against the customer's "over a week ago". It never knew the date.
This is Month 1's `failures.md` "known limitation", carried over.
**Fix:** the `agent` node appends `Today is <date>.` to every system prompt,
using `returns.today()` so tests can pin it. `chat.py`'s MCP prompt no
longer formats its own date.
**Tests:** `test_f4_system_prompt_carries_todays_date`.

### F5: "Good news" to an angry customer — open
**Seen (run 2):** the customer opens with "This is ridiculous… it STILL
isn't here". The reply starts "Good news: order 1048 qualifies for a
refund" with no acknowledgement or apology. The facts were right; the tone
was wrong.
**Not fixed:** it's a tone problem, and one sample isn't enough to tune a
prompt against. It belongs in Month 4's eval set as a rubric item
("acknowledges frustration before solving").

### F6: escalated to get a human's rejection overturned — fixed, verified live (run 4)
**Seen (run 3):** the reviewer rejected the refund. Claude then called
`escalate_to_human` "to get that actioned", and told the customer only that
"no refund has been issued", never that it was declined. That's an agent
routing around a human decision, which is the failure the approval gate
exists to prevent.
**Fix:** the rejection result the model reads now says the decision
stands: tell the customer plainly, don't retry, and don't escalate unless
the customer asks for a person. The judge was also wrong. It now expects
no return, no unrequested escalation, and a plain "not approved" after a
rejection.
**Tests:** `test_f6_*` (the message), and
`test_judge_catches_escalating_around_a_rejection` (run 3 replayed; the
judge must fail it).

## Supervisor, first live runs (2026-10-28)

`scripts/chat.py --supervisor --customer C-100`, same customer opening line
as the scenario. Two runs (`livecheck1`, `livecheck2`; the audit log and
`data/runs/` hold them). These are the first times Days 51-54 ran against
Claude.

### F7: a stale seed hid the order the scenario is about — fixed (data), cause open
**Seen (livecheck1):** the supervisor never mentioned order 1048; the
workers worked on 1042, found it "not late", and the flow never reached the
approval gate. The persisted profile in `data/memory.sqlite` was seeded on
Day 48 and `seed_profiles` never overwrites, so it never gained 1048 when
Day 50 added it. `scripts/e2e.py` doesn't show this: it seeds a fresh
in-memory store.
**Fix:** the stored C-100 profile was refreshed from `PROFILES` by hand
(notes untouched). **Open:** the same drift will recur for any profile
change. The CRM record probably shouldn't be a never-overwrite seed.

### F8: the supervisor path offers options nothing grounds, and acts without a confirmation — open
**Seen (livecheck1, livecheck2):** the order agent (which has no return
tool) proposed "refuse-on-delivery / return-to-sender" in its handoff note
before anything said it was possible; in livecheck1 the returns agent
retracted it ("please disregard that option") and a later reply promised
"I'll email you as soon as delivery registers", a follow-up nothing
performs (F3 again, from the supervisor's reply). In livecheck2 the final
reply offered "you can refuse delivery" *after the refund was rejected*,
when no return existed. (The same phrase in the single-agent demo recording
is grounded: `create_return`'s own `next_step` says it.)
Separately, in livecheck2 the returns agent called `create_return` on the
customer's first message, before the "Yes, refund it" the single-agent
flow waits for. The human gate caught it, but the confirmation rule in the
worker prompt didn't hold across the brief.
**Not fixed.** One more sample of what the README says: every hop is a
chance to lose a constraint. Candidates for next: put the confirmation rule
and "only offer what a tool's result says" in the supervisor's prompt too,
and add both as Month 4 rubric items.

### F9: first measured costs for the supervisor path
livecheck2 (to the approval gate, rejected by mistake, see below): 7 model
calls, $0.1880, 50s. livecheck1 (three customer turns): 16 calls, $0.3372,
90s. The single agent's e2e run 4 for comparison: 5 calls, $0.1156, 21s.
Different customer turns, so this is indicative, not a controlled
comparison, but 7 vs 5 calls to the same decision point agrees with the
README's estimate. (livecheck2's reviewer rejection was a piping mistake:
the scripted stdin fed "Yes. Refund it." to the `approve?` prompt.)

## Known gaps (not failures, carry forward)

- **No triage node or order specialist.** One ReAct agent plays every
  stage. The judge checks the stages as behaviour in the trace, so it
  still applies after Days 51–52 split the work into agents.
- **The scenario runs on the in-memory data, not mcp-commerce.**
  mcp-commerce's `get_order_status` has no delivery date or categories, so
  the return rules can't run on it (Day 47).
- **Order ownership isn't enforced.** "Only discuss these orders" is a
  prompt instruction (Day 48).
- **Made-up product facts.** On Day 48 Claude said "runs true to size" and
  "size up a half", with no tool backing either.
- **The audit trail skips input steps.** The customer's message entering
  state isn't a row (Day 49).
- **A fixed script.** In run 1, Claude asked "1048 or both?" and the
  scripted reply was "Yes. Refund it." Claude reasonably took that to mean
  1048. A scripted customer can't answer follow-up questions; Month 4's
  simulated user can.

## Seed set for Month 4 (`agent-evals`)

The scenario and its five-stage judge, as-is, plus the reject variant. Add
rubric items for F5 (tone) and F3 (no invented commitments), which this
judge doesn't score.
