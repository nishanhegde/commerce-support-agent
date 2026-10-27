"""Day 47: returns. Eligibility rules in plain Python, three tools on top.

The split that matters: the model decides *when* to check a return, never
*whether* one is allowed. `eligibility()` is an ordinary function with no
model in it. Ask it the same question twice and you get the same answer, it
runs in a test without an API key, and changing a rule is a code review
rather than a prompt tweak. SAP Commerce makes the same split: whether an
order is returnable is a chain of check strategies on the server, and the
storefront only asks.

The rules, in the order they're applied:

    1. The order and the line must exist.                 -> found: false
    2. The order must be delivered. A pending order is     -> stops here, with
       cancelled, not returned; an in-transit one hasn't      what to do instead
       arrived yet.
    3. Category: some categories can never go back         -> each failing rule
       (hygiene). Some have a shorter window.                 adds a reason;
    4. Window: N days from *delivery*, not from order.        all are reported
       30 by default, 10 for electronics.                     together
    5. Quantity: no more than was bought, minus what's
       already on an open return.

Rules 1-2 stop early because the later ones can't be evaluated without
them (no delivery date, no window). Rules 3-5 are collected, so a
customer who is both late and over quantity hears both at once instead of
fixing one and then hitting the next.

Returns are an OMS concern, not a storefront one, so these tools read the
in-memory ORDERS rather than mcp-commerce: its get_order_status has no
delivery date and no categories. Day 50's end-to-end run is where that gap
has to be closed.
"""

import itertools
from datetime import date, timedelta
from typing import Literal

from langchain_core.tools import tool

from support_agent.tools import ORDERS

DEFAULT_WINDOW_DAYS = 30
# Category -> days. Anything not listed gets DEFAULT_WINDOW_DAYS.
WINDOW_DAYS = {"electronics": 10}
# Category -> the reason a customer is given. Never returnable, at any age.
NON_RETURNABLE = {"foot-care": "insoles and foot-care items can't be returned for hygiene reasons"}

# Open returns, by return id. Module-level and in-memory, like ORDERS; the
# test conftest resets it.
RETURNS: dict[str, dict] = {}
ESCALATIONS: list[dict] = []

# A process-wide counter. itertools.count() is an endless iterator; next()
# on it is the Python spelling of an AtomicInteger's getAndIncrement (it is
# not thread-safe, which is fine for one CLI process).
_return_seq = itertools.count(1)
_ticket_seq = itertools.count(1)


def today() -> date:
    """The current date. A function, not a direct date.today() call, so tests
    can monkeypatch it: they patch `returns.today`, the name the tools look up
    at call time."""
    return date.today()


def already_returning(order_id: str, sku: str) -> int:
    return sum(
        r["quantity"] for r in RETURNS.values() if r["order_id"] == order_id and r["sku"] == sku
    )


def eligibility(order: dict | None, sku: str, quantity: int, on: date) -> dict:
    """Can `quantity` of `sku` from this order go back, as of `on`?

    Pure: everything it reads is an argument, apart from RETURNS (what's
    already being returned). Returns a dict because it goes straight to the
    model as a tool result.
    """
    if order is None:
        return {"found": False, "why": "no such order"}
    line = next((i for i in order["items"] if i["sku"] == sku), None)
    if line is None:
        skus = [i["sku"] for i in order["items"]]
        return {"found": False, "why": f"order {order['order_id']} has no item {sku}; it has {skus}"}

    result = {"order_id": order["order_id"], "sku": sku, "name": line["name"]}

    if order["status"] != "delivered":
        instead = {
            "pending": "it hasn't shipped, so it can be cancelled instead",
            "in_transit": "it hasn't been delivered yet; a return can start once it arrives",
        }.get(order["status"], "only delivered orders can be returned")
        return {**result, "eligible": False, "reasons": [f"order is {order['status']}: {instead}"]}

    reasons = []
    category = line["category"]
    if category in NON_RETURNABLE:
        reasons.append(NON_RETURNABLE[category])

    window = WINDOW_DAYS.get(category, DEFAULT_WINDOW_DAYS)
    delivered = date.fromisoformat(order["delivered_on"])
    # The last day is inclusive: delivered on the 1st with 30 days, the 31st
    # is still fine.
    return_by = delivered + timedelta(days=window)
    if on > return_by:
        reasons.append(
            f"{category} can be returned within {window} days of delivery; "
            f"this was delivered {delivered.isoformat()}, so the window closed {return_by.isoformat()}"
        )

    open_qty = already_returning(order["order_id"], sku)
    returnable_qty = line["qty"] - open_qty
    if quantity < 1:
        reasons.append("quantity must be at least 1")
    elif quantity > returnable_qty:
        already = f" ({open_qty} already on an open return)" if open_qty else ""
        reasons.append(f"only {returnable_qty} of {line['qty']} can be returned{already}")

    result["eligible"] = not reasons
    if reasons:
        result["reasons"] = reasons
    else:
        result["return_by"] = return_by.isoformat()
        result["refund_amount"] = quantity * line["unit_price"]
    return result


@tool
def check_return_eligibility(order_id: str, sku: str, quantity: int = 1) -> dict:
    """Check whether an item from an order can be returned. Read-only.

    Call this before offering or creating a return, and tell the customer the
    result and the reasons in plain words. If `eligible` is false, the reasons
    are final policy: don't offer the return anyway. If the order has several
    items and the customer hasn't said which, check each one.

    Args:
        order_id: The order number, digits only.
        sku: The item's SKU, as get_order_status lists it.
        quantity: How many units the customer wants to return.
    """
    return eligibility(ORDERS.get(order_id), sku, quantity, today())


@tool
def create_return(order_id: str, sku: str, quantity: int, reason: str) -> dict:
    """Open a return for an item and issue a return id. The refund is paid
    to the original payment method once the item arrives back and passes
    inspection, not now.

    Call this only after check_return_eligibility said eligible AND the
    customer has confirmed they want to go ahead. A human approves each call
    before it runs, so a result you see is final; a "rejected by a human
    reviewer" error means no return was created.

    Args:
        order_id: The order number, digits only.
        sku: The item's SKU.
        quantity: How many units are going back.
        reason: The customer's reason, in a few words.
    """
    # Check again, here. The docstring asks the model to check first, but a
    # docstring is a request, not a guarantee: this line is the guarantee.
    # It also covers the gap between the check and the approval, which can
    # be hours if the thread sat paused.
    check = eligibility(ORDERS.get(order_id), sku, quantity, today())
    if not check.get("eligible"):
        return {"created": False, **check}

    return_id = f"RMA-{order_id}-{next(_return_seq)}"
    RETURNS[return_id] = {
        "return_id": return_id,
        "order_id": order_id,
        "sku": sku,
        "quantity": quantity,
        "reason": reason,
        "status": "awaiting_item",
        "refund_amount": check["refund_amount"],
    }
    return {
        "created": True,
        **RETURNS[return_id],
        "next_step": f"ship the item back by {check['return_by']}; refund follows inspection",
    }


@tool
def escalate_to_human(summary: str, reason: str, priority: Literal["normal", "urgent"] = "normal") -> dict:
    """Hand the conversation to a human support agent. Stub: it opens a ticket
    and returns its id; nobody is actually paged yet.

    Call this when the customer asks for a person, when they dispute a policy
    answer you can't change, or when they need something your tools can't do.
    Give the customer the ticket id. Don't promise a response time.

    Args:
        summary: What the customer wants and what you've already checked, so
            the human doesn't have to ask again.
        reason: Why this needs a human, in a few words.
        priority: "urgent" only for a safety issue, suspected fraud, or a
            customer who has escalated before about the same problem.
    """
    ticket = {
        "ticket_id": f"ESC-{next(_ticket_seq):04d}",
        "queue": "support-tier2",
        "status": "open",
        "priority": priority,
        "reason": reason,
        "summary": summary,
    }
    ESCALATIONS.append(ticket)
    # The summary goes to the human, not back to the model, which wrote it.
    return {k: ticket[k] for k in ("ticket_id", "queue", "status", "priority")}


RETURN_TOOLS = [check_return_eligibility, create_return, escalate_to_human]
