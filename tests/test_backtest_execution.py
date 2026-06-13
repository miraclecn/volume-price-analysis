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


def test_execution_records_decision_date_for_no_next_bar_rejection():
    targets = pd.DataFrame(
        [{"trade_date": "2024-01-02", "portfolio_id": "p1", "code": "a", "target_weight": 0.5, "rank_n": 1, "trade_score": 0.9}]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "open": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
        ]
    )

    orders = simulate_rebalance_orders(targets, bars, pd.DataFrame(), 1000.0, ExecutionConfig())

    assert orders.iloc[0]["sim_date"] == "2024-01-02"
    assert orders.iloc[0]["status"] == "rejected"
    assert orders.iloc[0]["reason"] == "no_next_bar"


def test_execution_empty_daily_target_sells_current_positions():
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "open": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "a", "open": 11.0, "close": 11.0, "limit_up": 12.0, "limit_down": 10.0, "is_paused": False},
        ]
    )
    current_positions = pd.DataFrame([{"code": "a", "position_qty": 50.0}])

    orders = simulate_rebalance_orders(
        pd.DataFrame(),
        bars,
        current_positions,
        1000.0,
        ExecutionConfig(slippage_bps=0),
        decision_date="2024-01-02",
    )

    assert orders.iloc[0]["sim_date"] == "2024-01-03"
    assert orders.iloc[0]["side"] == "sell"
    assert orders.iloc[0]["qty"] == 50.0


def test_execution_does_not_rebalance_retained_holdings():
    targets = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "portfolio_id": "p1",
                "code": "a",
                "target_weight": 0.10,
                "signal_action": "hold",
            }
        ]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "open": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "a", "open": 20.0, "close": 20.0, "limit_up": 22.0, "limit_down": 18.0, "is_paused": False},
        ]
    )
    current_positions = pd.DataFrame([{"code": "a", "position_qty": 50.0}])

    orders = simulate_rebalance_orders(
        targets,
        bars,
        current_positions,
        1000.0,
        ExecutionConfig(slippage_bps=0),
        decision_date="2024-01-02",
    )

    assert orders.empty
