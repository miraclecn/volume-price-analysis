from __future__ import annotations

import pandas as pd

from ml_stock_selector.backtest.engine import BacktestConfig, run_backtest
from ml_stock_selector.backtest.execution import ExecutionConfig


def test_backtest_records_entry_and_holding_days_on_positions_and_sell_orders():
    targets = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "portfolio_id": "p1",
                "code": "a",
                "target_weight": 0.5,
                "rank_n": 1,
                "trade_score": 0.9,
                "entry_reason": "core_pool",
            },
            {
                "trade_date": "2024-01-04",
                "portfolio_id": "p1",
                "code": "a",
                "target_weight": 0.0,
                "rank_n": 1,
                "trade_score": 0.2,
                "exit_reason": "score_exit",
            },
        ]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "open": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "a", "open": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "a", "open": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-05", "code": "a", "open": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
        ]
    )

    result = run_backtest(
        targets,
        bars,
        BacktestConfig(1000.0, "p1", ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0)),
    )

    sell = result.orders[result.orders["side"] == "sell"].iloc[0]
    assert sell["entry_date"] == "2024-01-03"
    assert sell["exit_date"] == "2024-01-05"
    assert sell["holding_days"] == 2
    assert sell["exit_reason"] == "score_exit"
    assert result.positions["entry_date"].dropna().unique().tolist() == ["2024-01-03"]


def test_rejected_sell_records_blocked_reason_and_position_continues():
    targets = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "portfolio_id": "p1", "code": "a", "target_weight": 0.5, "rank_n": 1, "trade_score": 0.9},
            {
                "trade_date": "2024-01-03",
                "portfolio_id": "p1",
                "code": "a",
                "target_weight": 0.0,
                "rank_n": 1,
                "trade_score": 0.2,
                "exit_reason": "score_exit",
            },
        ]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "open": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "a", "open": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "a", "open": 9.0, "close": 9.0, "limit_up": 10.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-05", "code": "a", "open": 9.5, "close": 9.5, "limit_up": 10.5, "limit_down": 8.5, "is_paused": False},
        ]
    )

    result = run_backtest(
        targets,
        bars,
        BacktestConfig(1000.0, "p1", ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0)),
    )

    sell = result.orders[result.orders["side"] == "sell"].iloc[0]
    assert sell["status"] == "rejected"
    assert sell["sell_blocked_reason"] == "limit_down"
    assert result.positions[result.positions["sim_date"] == "2024-01-05"]["code"].tolist() == ["a"]
