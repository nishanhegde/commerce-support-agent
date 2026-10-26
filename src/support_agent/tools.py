"""Day 42: two tools, as LangChain `@tool` functions. Day 44 adds a third,
`cancel_order`, the first one that changes anything.

Compare with Month 1's tools.py: there, a tool was a hand-written JSON schema
dict plus a separate entry in an IMPLEMENTATIONS map. Here `@tool` derives the
schema from the signature and type hints, and the description from the
docstring — one object holds name, schema, description, and the callable.

The data is in-memory on purpose. Week 10 swaps these for the `mcp-commerce`
server; today the only new thing should be the graph.
"""

from langchain_core.tools import tool

ORDERS = {
    "1042": {
        "order_id": "1042",
        "status": "in_transit",
        "carrier": "BlueDart",
        "tracking_number": "BD-99120384",
        "estimated_delivery": "2026-10-28",
        "items": [{"sku": "RS-4471-BLK", "name": "Trail Runner 4", "qty": 1}],
    },
    "1043": {
        "order_id": "1043",
        "status": "delivered",
        "carrier": "Delhivery",
        "tracking_number": "DL-40028871",
        "delivered_on": "2026-10-20",
        "items": [{"sku": "JK-2210-GRN", "name": "Stormline Jacket", "qty": 1}],
    },
    "1044": {
        "order_id": "1044",
        "status": "pending",
        "carrier": None,
        "tracking_number": None,
        "items": [{"sku": "AC-0031-BLK", "name": "Cushion Insole", "qty": 2}],
    },
}

PRODUCTS = [
    {"sku": "RS-4471-BLK", "name": "Trail Runner 4", "price": 7999, "in_stock": True},
    {"sku": "RS-3020-BLU", "name": "Road Runner 2", "price": 4499, "in_stock": True},
    {"sku": "JK-2210-GRN", "name": "Stormline Jacket", "price": 5999, "in_stock": False},
    {"sku": "AC-0031-BLK", "name": "Cushion Insole", "price": 799, "in_stock": True},
]


@tool
def get_order_status(order_id: str) -> dict:
    """Look up the current shipping status, carrier, and tracking number for one order.

    Call this whenever the customer asks where an order is, whether it has
    shipped, or when it will arrive, and you have an order number. Never guess
    at an order's status.

    Args:
        order_id: The order number, digits only.
    """
    order = ORDERS.get(order_id)
    # A missing order is a business fact to relay, not a tool failure, so it
    # comes back as data rather than an exception. Same split as Month 1.
    return order if order else {"found": False, "order_id": order_id}


@tool
def search_products(keywords: list[str], max_price: int | None = None) -> list[dict]:
    """Search the product catalogue by keyword, optionally under a price ceiling.

    Call this when the customer asks what you sell, wants something cheaper or
    similar, or asks whether something is in stock. Prices are in rupees.

    Args:
        keywords: Search terms drawn from the customer's own words.
        max_price: Price ceiling in rupees. Set only if the customer stated a number.
    """
    words = [k.lower() for k in keywords]
    return [
        p
        for p in PRODUCTS
        if any(w in p["name"].lower() for w in words)
        and (max_price is None or p["price"] <= max_price)
    ]


@tool
def cancel_order(order_id: str, reason: str) -> dict:
    """Cancel an order that has not shipped yet. This cannot be undone.

    Call this only when the customer has clearly asked to cancel a specific
    order. A human approves or rejects each call before it runs, so the
    review is already over when you see the result: `cancelled: true` is
    final, and a "rejected by a human reviewer" error means the order was
    not cancelled.

    Args:
        order_id: The order number, digits only.
        reason: The customer's reason for cancelling, in a few words.
    """
    order = ORDERS.get(order_id)
    if order is None:
        return {"found": False, "order_id": order_id}
    # Shipped or delivered orders go through returns, not cancellation. Like
    # a missing order, that's a business answer, so it's data, not an error.
    if order["status"] != "pending":
        return {
            "cancelled": False,
            "order_id": order_id,
            "why": f"order is {order['status']}; only pending orders can be cancelled",
        }
    order["status"] = "cancelled"
    order["cancel_reason"] = reason
    return {"cancelled": True, "order_id": order_id}


TOOLS = [get_order_status, search_products, cancel_order]

# Tools the graph will not run without a human saying yes (Day 44). A set of
# names, not a flag on the tool, so the policy lives in one place you can read.
NEEDS_APPROVAL = {"cancel_order"}
