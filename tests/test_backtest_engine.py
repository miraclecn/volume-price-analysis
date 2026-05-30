from __future__ import annotations

import pandas as pd

from ml_stock_selector.backtest.engine import BacktestConfig, run_backtest
from ml_stock_selector.backtest.execution import ExecutionConfig


def test_backtest_outputs_orders_positions_and_nav():
    targets = pd.DataFrame(
        [{"trade_date": "2024-01-02", "portfolio_id": "p1", "code": "a", "target_weight": 0.5, "rank_n": 1, "trade_score": 0.9}]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "a", "open": 10.1, "high": 10.3, "low": 10.0, "close": 10.2, "limit_up": 11.2, "limit_down": 9.2, "is_paused": False},
        ]
    )

    result = run_backtest(targets, bars, BacktestConfig(1000.0, "p1", ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0)))

    assert not result.orders.empty
    assert not result.positions.empty
    assert result.nav.iloc[-1]["nav"] > 0

