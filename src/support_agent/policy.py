"""Day 49: the policy layer. Which tool calls a human must approve, and why.

Day 44 answered that with a set of names: {"cancel_order", "create_return"}.
A name can't express "refunds over ₹5,000", because the answer depends on
the call's arguments, and on the order behind them. So a policy is now a
function from one tool call to a Verdict:

    policy(call) -> None                 not an action: lookups, notes, tickets
                 -> Verdict(human=False)  an action the rules let through
                 -> Verdict(human=True)   an action that waits for a person

Both kinds of Verdict are logged as approvals (react.py): a refund the
policy let through was still approved, by a rule rather than a person, and
an auditor will ask which rule.

The refund amount comes from returns.eligibility(), the same function
create_return runs, not from anything the model says. The model never
states an amount, so it can't understate one to slip under the limit.
"""

from collections.abc import Callable
from dataclasses import dataclass

from support_agent import returns
from support_agent.tools import ORDERS

# Refunds above this many rupees wait for a person. At or below, they go
# straight through.
REFUND_APPROVAL_THRESHOLD = 5000


@dataclass(frozen=True)
class Verdict:
    human: bool  # True: interrupt and wait for a reviewer
    rule: str  # a stable id, for the log: "refund_over_threshold"
    why: str  # one line for the reviewer and the log


Policy = Callable[[dict], Verdict | None]


def default_policy(call: dict) -> Verdict | None:
    """The in-memory tools' rules."""
    name, args = call["name"], call["args"]
    if name == "cancel_order":
        # Day 50 fix (failures.md F2): only a cancel that can happen is put to
        # a human. Same reasoning as an ineligible return below.
        order = ORDERS.get(args.get("order_id", ""))
        if order is None or order["status"] != "pending":
            return Verdict(False, "cancel_ineligible", "not pending; the tool will refuse it")
        return Verdict(True, "cancel_always_reviewed", "cancellation can't be undone")
    if name == "create_return":
        return _refund_verdict(args)
    return None


def _refund_verdict(args: dict) -> Verdict:
    check = returns.eligibility(
        ORDERS.get(args.get("order_id", "")),
        args.get("sku", ""),
        args.get("quantity", 0),
        returns.today(),
    )
    if not check.get("eligible"):
        # Nothing to approve: create_return will refuse it on its own, and
        # asking a human to approve a refusal wastes their time.
        return Verdict(False, "return_ineligible", "not eligible; the tool will refuse it")
    amount = check["refund_amount"]
    if amount > REFUND_APPROVAL_THRESHOLD:
        return Verdict(
            True,
            "refund_over_threshold",
            f"refund ₹{amount:,} is over the ₹{REFUND_APPROVAL_THRESHOLD:,} limit",
        )
    return Verdict(
        False,
        "refund_within_threshold",
        f"refund ₹{amount:,} is within the ₹{REFUND_APPROVAL_THRESHOLD:,} limit",
    )


def by_name(names: set[str]) -> Policy:
    """Day 44's rule as a policy: these tools always need a human. Used for
    mcp-commerce, whose write tools come from the server's annotations."""

    def policy(call: dict) -> Verdict | None:
        if call["name"] in names:
            return Verdict(True, "write_tool", "the server marks this tool as not read-only")
        return None

    return policy
