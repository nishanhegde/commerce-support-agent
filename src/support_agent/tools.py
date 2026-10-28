"""Day 42: two tools, as LangChain `@tool` functions. Day 44 adds a third,
`cancel_order`, the first one that changes anything.

Compare with Month 1's tools.py: there, a tool was a hand-written JSON schema
dict plus a separate entry in an IMPLEMENTATIONS map. Here `@tool` derives the
schema from the signature and type hints, and the description from the
docstring — one object holds name, schema, description, and the callable.

The data is in-memory on purpose. Week 10 swaps these for the `mcp-commerce`
server; today the only new thing should be the graph.

Day 47's return tools live in returns.py, over the same ORDERS.
"""

from langchain_core.tools import tool

from support_agent.resilience import mark_read_only

# Day 47: each line carries its category and the price paid. The price is
# copied onto the line at order time (OCC's entry basePrice), not looked up
# from the catalogue: a refund is for what the customer paid, not today's price.
ORDERS = {
    "1042": {
        "order_id": "1042",
        "status": "in_transit",
        "carrier": "BlueDart",
        "tracking_number": "BD-99120384",
        "estimated_delivery": "2026-10-28",
        "items": [
            {"sku": "RS-4471-BLK", "name": "Trail Runner 4", "qty": 1,
             "category": "footwear", "unit_price": 7999},
        ],
    },
    "1043": {
        "order_id": "1043",
        "status": "delivered",
        "carrier": "Delhivery",
        "tracking_number": "DL-40028871",
        "delivered_on": "2026-10-20",
        "items": [
            {"sku": "JK-2210-GRN", "name": "Stormline Jacket", "qty": 1,
             "category": "apparel", "unit_price": 5999},
        ],
    },
    "1044": {
        "order_id": "1044",
        "status": "pending",
        "carrier": None,
        "tracking_number": None,
        "items": [
            {"sku": "AC-0031-BLK", "name": "Cushion Insole", "qty": 2,
             "category": "foot-care", "unit_price": 799},
        ],
    },
    # Day 47: delivered orders that each hit a different return rule.
    # Delivered well over 30 days ago: outside the standard window.
    "1045": {
        "order_id": "1045",
        "status": "delivered",
        "carrier": "BlueDart",
        "tracking_number": "BD-99108812",
        "delivered_on": "2026-09-10",
        "items": [
            {"sku": "RS-3020-BLU", "name": "Road Runner 2", "qty": 1,
             "category": "footwear", "unit_price": 4499},
        ],
    },
    # A mixed basket: the shoes can go back, the insoles can't.
    "1046": {
        "order_id": "1046",
        "status": "delivered",
        "carrier": "Delhivery",
        "tracking_number": "DL-40031190",
        "delivered_on": "2026-10-22",
        "items": [
            {"sku": "RS-3020-BLU", "name": "Road Runner 2", "qty": 2,
             "category": "footwear", "unit_price": 4499},
            {"sku": "AC-0031-BLK", "name": "Cushion Insole", "qty": 1,
             "category": "foot-care", "unit_price": 799},
        ],
    },
    # Inside 30 days, but electronics get 10.
    "1047": {
        "order_id": "1047",
        "status": "delivered",
        "carrier": "BlueDart",
        "tracking_number": "BD-99117735",
        "delivered_on": "2026-10-12",
        "items": [
            {"sku": "EL-0900-BLK", "name": "Pulse GPS Watch", "qty": 1,
             "category": "electronics", "unit_price": 12999},
        ],
    },
    # Day 50: the end-to-end scenario's order. Shipped, and 9 days past its
    # estimate on 2026-10-27. Over ₹5,000, so any refund needs a human.
    "1048": {
        "order_id": "1048",
        "status": "in_transit",
        "carrier": "BlueDart",
        "tracking_number": "BD-99110457",
        "estimated_delivery": "2026-10-18",
        "items": [
            {"sku": "RS-4471-BLK", "name": "Trail Runner 4", "qty": 1,
             "category": "footwear", "unit_price": 7999},
        ],
    },
}

PRODUCTS = [
    {"sku": "RS-4471-BLK", "name": "Trail Runner 4", "price": 7999, "in_stock": True},
    {"sku": "RS-3020-BLU", "name": "Road Runner 2", "price": 4499, "in_stock": True},
    {"sku": "JK-2210-GRN", "name": "Stormline Jacket", "price": 5999, "in_stock": False},
    {"sku": "AC-0031-BLK", "name": "Cushion Insole", "price": 799, "in_stock": True},
    {"sku": "EL-0900-BLK", "name": "Pulse GPS Watch", "price": 12999, "in_stock": True},
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

# Day 56: the lookups are safe to repeat, so a failed call is retried.
mark_read_only(get_order_status, search_products)

# Which of these need a human's approval lived here as a set of names from
# Day 44. Day 49 moved it to policy.py, where it can depend on the arguments.
