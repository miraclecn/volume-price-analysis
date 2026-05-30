from __future__ import annotations

import pandas as pd

from ml_stock_selector.backtest.execution import ExecutionConfig, simulate_rebalance_orders


def test_execution_is_t_plus_one_and_rejects_limit_up_buy():
    targets = pd.DataFrame(
        [{"trade_date": "2024-01-02", "portfolio_id": "p1", "code": "a", "target_weight": 0.5, "rank_n": 1, "trade_score": 0.9}]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "open": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "a", "open": 11.0, "close": 11.0, "limit_up": 11.0, "limit_down": 10.0, "is_paused": False},
        ]
    )

    orders = simulate_rebalance_orders(targets, bars, pd.DataFrame(), 1000.0, ExecutionConfig())

    assert orders.iloc[0]["sim_date"] == "2024-01-03"
    assert orders.iloc[0]["status"] == "rejected"
    assert orders.iloc[0]["reason"] == "limit_up"

