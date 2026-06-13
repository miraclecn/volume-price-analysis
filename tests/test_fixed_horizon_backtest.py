from __future__ import annotations

import pandas as pd

from ml_stock_selector.backtest.engine import BacktestConfig, run_fixed_horizon_backtest
from ml_stock_selector.backtest.execution import ExecutionConfig
from ml_stock_selector.portfolio.constraints import FixedHorizonRiskFilterConfig


def _bars() -> pd.DataFrame:
    rows = []
    for date in pd.bdate_range("2024-01-02", periods=9).strftime("%Y-%m-%d"):
        rows.append(
            {
                "trade_date": date,
                "code": "000001.SZ",
                "open": 10.0,
                "close": 10.0,
                "limit_up": 11.0,
                "limit_down": 9.0,
                "is_paused": False,
            }
        )
    return pd.DataFrame(rows)


def _scored(risk_by_date: dict[str, float] | None = None) -> pd.DataFrame:
    risk_by_date = risk_by_date or {}
    return pd.DataFrame(
        [
            {
                "trade_date": date,
                "code": "000001.SZ",
                "absolute_rank_pct": 0.95 if i == 0 else 0.01,
                "risk_rank_pct": risk_by_date.get(date, 0.10),
                "is_bse": False,
                "is_st": False,
                "is_paused": False,
                "can_buy_next_open": True,
                "can_sell_next_open": True,
                "adv20_amount": 100_000_000,
            }
            for i, date in enumerate(pd.bdate_range("2024-01-02", periods=8).strftime("%Y-%m-%d"))
        ]
    )


def _constraints(**overrides):
    values = {
        "target_positions": 1,
        "hard_max_positions": 1,
        "max_initial_entries": 1,
        "max_new_entries_per_day": 1,
        "min_abs_rank_pct": 0.70,
        "risk_entry_max_rank_pct": 0.55,
        "risk_exit_rank_pct": 0.85,
        "min_adv20_amount": 50_000_000,
        "min_position_weight": 1.0,
        "max_position_weight": 1.0,
    }
    values.update(overrides)
    return FixedHorizonRiskFilterConfig(**values)


def test_fixed_horizon_backtest_holds_until_fifth_trading_day_without_score_exit():
    result = run_fixed_horizon_backtest(
        _scored(),
        _bars(),
        _constraints(),
        BacktestConfig(1000.0, "fixed", ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0)),
        run_id="run",
        fold_id="fold",
    )

    sells = result.orders[(result.orders["side"] == "sell") & (result.orders["status"] == "filled")]
    assert sells["exit_reason"].tolist() == ["time_exit"]
    assert sells.iloc[0]["holding_days"] == 5
    assert "score_exit" not in set(result.orders["exit_reason"].dropna())
    assert result.orders["strategy_id"].eq("abs_ranker_fixed_5d_risk_filter_v1").all()


def test_fixed_horizon_backtest_risk_exit_can_sell_early_and_no_risk_variant_cannot():
    risk_exit = run_fixed_horizon_backtest(
        _scored({"2024-01-04": 0.95}),
        _bars(),
        _constraints(),
        BacktestConfig(1000.0, "fixed", ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0)),
    )
    no_risk_exit = run_fixed_horizon_backtest(
        _scored({"2024-01-04": 0.95}),
        _bars(),
        _constraints(strategy_id="abs_ranker_fixed_5d_no_risk_exit_v1", enable_risk_exit=False),
        BacktestConfig(1000.0, "fixed", ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0)),
    )

    risk_sell = risk_exit.orders[(risk_exit.orders["side"] == "sell") & (risk_exit.orders["status"] == "filled")].iloc[0]
    no_risk_sell = no_risk_exit.orders[(no_risk_exit.orders["side"] == "sell") & (no_risk_exit.orders["status"] == "filled")].iloc[0]
    assert risk_sell["exit_reason"] == "risk_exit"
    assert risk_sell["holding_days"] < 5
    assert no_risk_sell["exit_reason"] == "time_exit"
