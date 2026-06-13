from __future__ import annotations

import pandas as pd

from ml_stock_selector.backtest.engine import BacktestConfig, run_backtest, run_holding_aware_backtest
from ml_stock_selector.backtest.execution import ExecutionConfig
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.holding_policy import HoldingPolicy
from ml_stock_selector.portfolio.constructor import PORTFOLIO_DIAGNOSTICS_ATTR


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
    assert result.orders["order_seq"].tolist() == list(range(1, len(result.orders) + 1))


def test_backtest_marks_nav_daily_after_entry_and_rebalances_to_target_weight():
    targets = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "portfolio_id": "p1", "code": "a", "target_weight": 0.5, "rank_n": 1, "trade_score": 0.9},
            {"trade_date": "2024-01-03", "portfolio_id": "p1", "code": "a", "target_weight": 0.5, "rank_n": 1, "trade_score": 0.9},
        ]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "a", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "a", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-05", "code": "a", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
        ]
    )

    result = run_backtest(targets, bars, BacktestConfig(1000.0, "p1", ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0)))

    assert result.nav["sim_date"].tolist() == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    assert result.nav.iloc[-1]["gross_exposure"] == 0.5
    assert result.positions[result.positions["sim_date"] == "2024-01-05"].iloc[0]["position_qty"] == 50.0


def test_backtest_empty_daily_decision_liquidates_previous_holdings():
    targets = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "portfolio_id": "p1", "code": "a", "target_weight": 0.5, "rank_n": 1, "trade_score": 0.9},
        ]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "a", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "a", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
        ]
    )

    result = run_backtest(
        targets,
        bars,
        BacktestConfig(
            1000.0,
            "p1",
            ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0),
            decision_dates=["2024-01-02", "2024-01-03"],
        ),
    )

    assert result.orders["side"].tolist() == ["buy", "sell"]
    assert result.nav.set_index("sim_date").loc["2024-01-04", "gross_exposure"] == 0.0


def test_backtest_scales_buys_to_available_cash_and_never_goes_negative():
    targets = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "portfolio_id": "p1", "code": "a", "target_weight": 0.8, "rank_n": 1, "trade_score": 0.9},
            {"trade_date": "2024-01-03", "portfolio_id": "p1", "code": "a", "target_weight": 0.8, "rank_n": 1, "trade_score": 0.9, "signal_action": "hold"},
            {"trade_date": "2024-01-03", "portfolio_id": "p1", "code": "b", "target_weight": 0.8, "rank_n": 2, "trade_score": 0.8},
        ]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "a", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "b", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "a", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "b", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
        ]
    )

    result = run_backtest(
        targets,
        bars,
        BacktestConfig(1000.0, "p1", ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0)),
    )

    day4 = result.nav.set_index("sim_date").loc["2024-01-04"]
    b_buy = result.orders[(result.orders["code"] == "b") & (result.orders["side"] == "buy")].iloc[0]
    assert b_buy["qty"] == 20.0
    assert day4["cash"] == 0.0
    assert day4["gross_exposure"] == 1.0
    assert result.nav["cash"].min() >= 0.0


def test_backtest_does_not_duplicate_pending_sell_orders_when_next_bar_is_delayed():
    targets = pd.DataFrame(
        [
            {"trade_date": "2024-01-01", "portfolio_id": "p1", "code": "a", "target_weight": 0.5, "rank_n": 1, "trade_score": 0.9},
        ]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-01", "code": "a", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-02", "code": "a", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "b", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "a", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
        ]
    )

    result = run_backtest(
        targets,
        bars,
        BacktestConfig(
            1000.0,
            "p1",
            ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0),
            decision_dates=["2024-01-01", "2024-01-02", "2024-01-03"],
        ),
    )

    assert result.orders["side"].tolist() == ["buy", "sell"]
    assert result.nav.set_index("sim_date").loc["2024-01-04", "gross_exposure"] == 0.0


def test_backtest_result_carries_portfolio_construction_diagnostics():
    targets = pd.DataFrame(
        [{"trade_date": "2024-01-02", "portfolio_id": "p1", "code": "a", "target_weight": 0.5, "rank_n": 1, "trade_score": 0.9}]
    )
    targets.attrs[PORTFOLIO_DIAGNOSTICS_ATTR] = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "run_id": "run",
                "fold_id": "wf_2020",
                "portfolio_id": "p1",
                "score_version": "v2_three_model",
                "final_selected_count": 1,
            }
        ]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "a", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
        ]
    )

    result = run_backtest(
        targets,
        bars,
        BacktestConfig(1000.0, "p1", ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0)),
    )

    assert result.portfolio_diagnostics["final_selected_count"].tolist() == [1]


def test_holding_aware_backtest_constructs_daily_targets_from_live_holdings():
    scored = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "a",
                "industry_code": "I1",
                "industry_name": "Industry 1",
                "trade_score_v2": 0.90,
                "absolute_rank_pct": 0.90,
                "active_rank_pct": 0.90,
                "risk_rank_pct": 0.10,
                "risk_prob": 0.10,
                "is_st": False,
                "is_paused": False,
                "is_bse": False,
                "can_buy_next_open": True,
                "can_sell_next_open": True,
                "adv20_amount": 100.0,
            },
            {
                "trade_date": "2024-01-03",
                "code": "b",
                "industry_code": "I2",
                "industry_name": "Industry 2",
                "trade_score_v2": 0.90,
                "absolute_rank_pct": 0.90,
                "active_rank_pct": 0.90,
                "risk_rank_pct": 0.10,
                "risk_prob": 0.10,
                "is_st": False,
                "is_paused": False,
                "is_bse": False,
                "can_buy_next_open": True,
                "can_sell_next_open": True,
                "adv20_amount": 100.0,
            },
        ]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "open": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "a", "open": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "b", "open": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "a", "open": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "b", "open": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
        ]
    )

    result = run_holding_aware_backtest(
        scored,
        bars,
        PortfolioConstraints(
            target_positions=2,
            hard_max_positions=2,
            max_initial_entries=1,
            max_new_entries_per_day=1,
            max_industry_names=99,
            min_adv20_amount=0,
            candidate_min_trade_score=0.65,
            core_min_trade_score=0.65,
            holding_policy=HoldingPolicy(min_hold_days=3),
        ),
        BacktestConfig(1000.0, "p1", ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0)),
        min_weight=0.5,
        max_weight=0.5,
    )

    day4_positions = set(result.positions[result.positions["sim_date"] == "2024-01-04"]["code"])
    assert {"a", "b"}.issubset(day4_positions)
    assert (result.portfolio_diagnostics["hold_due_to_min_days_count"] > 0).any()
