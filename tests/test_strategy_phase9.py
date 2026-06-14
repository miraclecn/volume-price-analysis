from __future__ import annotations

import pandas as pd

from ml_stock_selector.strategy.allocation import allocate_strategy_sleeves
from ml_stock_selector.strategy.backtest import backtest_strategy_allocation
from ml_stock_selector.strategy.ensemble import default_phase9_sleeves
from ml_stock_selector.strategy.model_health import summarize_model_health_rows
from ml_stock_selector.strategy.regime import classify_market_regime
from ml_stock_selector.strategy.risk_budget import drawdown_multiplier, regime_weights


def test_phase9_terms_define_current_best_as_core_sleeve():
    sleeves = default_phase9_sleeves(core_bundle_id="bundle_core")
    core = next(sleeve for sleeve in sleeves if sleeve.sleeve == "core")

    assert core.strategy_id == "holding_aware_v2"
    assert core.score_version == "v2_absolute_risk_filter"
    assert core.experiment_family == "expanding_gap"
    assert core.gap_type == "one_year_gap"
    assert core.model_roles == ("absolute", "risk")
    assert core.bundle_id == "bundle_core"


def test_regime_and_drawdown_budget_rules_match_phase9_policy():
    assert classify_market_regime(
        trend_score=0.8,
        breadth_score=0.75,
        sentiment_score=0.7,
        liquidity_score=0.7,
        volatility_score=0.25,
    ) == "risk_on"
    assert regime_weights("risk_on") == {
        "core": 0.55,
        "aggressive": 0.25,
        "fixed_horizon": 0.10,
        "cash": 0.10,
    }
    assert regime_weights("risk_off")["cash"] == 0.70
    assert drawdown_multiplier(-0.04) == 1.0
    assert drawdown_multiplier(-0.08) == 0.75
    assert drawdown_multiplier(-0.12) == 0.50
    assert drawdown_multiplier(-0.18) == 0.25
    assert drawdown_multiplier(-0.25) == 0.0


def test_allocate_strategy_sleeves_moves_disabled_and_drawdown_budget_to_cash():
    sleeves = default_phase9_sleeves(core_bundle_id="core_bundle", aggressive_bundle_id="aggressive_bundle")

    rows = allocate_strategy_sleeves(
        trade_date="2026-06-12",
        sleeves=sleeves,
        final_regime="risk_on",
        account_drawdown=-0.12,
        health_enabled_by_bundle={"core_bundle": True, "aggressive_bundle": False},
        generated_at="t",
    )
    values = {
        row.sleeve: row.final_weight
        for row in rows.itertuples(index=False)
    }

    assert values["core"] == 0.275
    assert values["aggressive"] == 0.0
    assert values["fixed_horizon"] == 0.05
    assert values["cash"] == 0.675
    assert round(sum(values.values()), 12) == 1.0


def test_model_health_disables_bundle_after_large_rolling_drawdown():
    nav = pd.DataFrame(
        {
            "sim_date": [f"2026-01-{day:02d}" for day in range(1, 11)],
            "nav": [100, 104, 106, 105, 103, 102, 100, 90, 82, 78],
        }
    )

    rows = summarize_model_health_rows(
        nav,
        model_or_bundle_id="core_bundle",
        strategy_id="holding_aware_v2",
        score_version="v2_absolute_risk_filter",
        long_window=5,
        short_window=3,
        max_drawdown_threshold=-0.20,
    )
    latest = rows.iloc[-1]

    assert latest["rolling_60d_drawdown"] < -0.20
    assert latest["enabled_by_health"] is False
    assert latest["reason"] == "rolling_drawdown_breached"


def test_strategy_allocation_backtest_combines_sleeve_returns():
    allocation = pd.DataFrame(
        [
            {"trade_date": "2026-01-01", "strategy_id": "core_strategy", "sleeve": "core", "score_version": "v1", "final_weight": 0.50},
            {"trade_date": "2026-01-01", "strategy_id": "aggressive_strategy", "sleeve": "aggressive", "score_version": "v1", "final_weight": 0.30},
            {"trade_date": "2026-01-01", "strategy_id": "cash_reserve", "sleeve": "cash", "score_version": "cash", "final_weight": 0.20},
            {"trade_date": "2026-01-02", "strategy_id": "core_strategy", "sleeve": "core", "score_version": "v1", "final_weight": 0.50},
            {"trade_date": "2026-01-02", "strategy_id": "aggressive_strategy", "sleeve": "aggressive", "score_version": "v1", "final_weight": 0.30},
            {"trade_date": "2026-01-02", "strategy_id": "cash_reserve", "sleeve": "cash", "score_version": "cash", "final_weight": 0.20},
        ]
    )
    sleeve_nav = pd.DataFrame(
        [
            {"sim_date": "2026-01-01", "strategy_id": "core_strategy", "score_version": "v1", "nav": 1.0},
            {"sim_date": "2026-01-02", "strategy_id": "core_strategy", "score_version": "v1", "nav": 1.10},
            {"sim_date": "2026-01-01", "strategy_id": "aggressive_strategy", "score_version": "v1", "nav": 1.0},
            {"sim_date": "2026-01-02", "strategy_id": "aggressive_strategy", "score_version": "v1", "nav": 1.20},
        ]
    )

    nav = backtest_strategy_allocation(allocation, sleeve_nav, strategy_id="phase9_ensemble_v1")

    assert nav["strategy_id"].eq("phase9_ensemble_v1").all()
    assert nav["nav"].tolist() == [1.0, 1.11]
    assert nav["gross_exposure"].tolist() == [0.8, 0.8]
