import copy

import pytest

from support_agent import returns, tools


@pytest.fixture(autouse=True)
def fresh_orders():
    """cancel_order mutates the module-level ORDERS dict. Put it back after
    every test so one test's cancellation can't leak into the next.
    Day 47: same for the returns and escalations the return tools record."""
    saved = copy.deepcopy(tools.ORDERS)
    yield
    tools.ORDERS.clear()
    tools.ORDERS.update(saved)
    returns.RETURNS.clear()
    returns.ESCALATIONS.clear()
