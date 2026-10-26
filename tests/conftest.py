import copy

import pytest

from support_agent import tools


@pytest.fixture(autouse=True)
def fresh_orders():
    """cancel_order mutates the module-level ORDERS dict. Put it back after
    every test so one test's cancellation can't leak into the next."""
    saved = copy.deepcopy(tools.ORDERS)
    yield
    tools.ORDERS.clear()
    tools.ORDERS.update(saved)
