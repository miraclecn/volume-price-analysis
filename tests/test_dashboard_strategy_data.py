from __future__ import annotations

import pytest

from dashboard.strategy_data import (
    current_strategy_config,
    current_strategy_model_summary,
    current_strategy_report,
    strategy_report_paths,
)


def test_current_strategy_config_exposes_live_risk_controls():
    config = current_strategy_config()

    assert config["portfolio_id"] == "mkt_tier_profit_protect"
    assert config["score_version"] == "v2_alpha_ret5d_fund_fixed_a160_r120_20260621"
    assert config["initial_cash"] == pytest.approx(1_000_000.0)
    assert config["execution"]["slippage_bps"] == pytest.approx(10.0)
    assert config["execution"]["commission_bps"] == pytest.approx(3.0)
    assert config["execution"]["stamp_duty_bps"] == pytest.approx(5.0)
    assert config["profit_protect_enabled"] is True
    assert config["profit_protect_min_days"] == 3
    assert config["profit_protect_min_gain"] == pytest.approx(0.03)
    assert config["profit_protect_exit_below"] == pytest.approx(0.005)
    assert config["market_zero_below"] == pytest.approx(0.375)
    assert config["market_half_below"] == pytest.approx(0.475)
    assert config["constraints"]["target_positions"] == 12
    assert config["constraints"]["hard_max_positions"] == 15
    assert config["constraints"]["max_new_entries_per_day"] == 4
    assert config["constraints"]["candidate_risk_max_rank_pct"] == pytest.approx(0.55)
    assert config["constraints"]["core_risk_max_rank_pct"] == pytest.approx(0.45)
    assert config["constraints"]["holding_policy"]["risk_exit_rank_pct"] == pytest.approx(0.75)
    assert config["constraints"]["holding_policy"]["risk_exit_prob"] == pytest.approx(0.60)


def test_current_strategy_report_loads_curves_metrics_and_attribution():
    report = current_strategy_report()

    assert report.summary.iloc[0]["variant"] == "mkt_tier_profit_protect"
    assert report.summary.iloc[0]["total_return"] > 10
    assert "sharpe" in report.summary.columns
    assert "sortino" in report.summary.columns
    assert "volatility" in report.summary.columns
    assert "calmar" in report.summary.columns
    assert not report.nav.empty
    assert {"sim_date", "nav", "drawdown", "daily_return"}.issubset(report.nav.columns)
    assert report.nav["drawdown"].min() < -0.15
    assert not report.yearly.empty
    assert not report.exit_attribution.empty
    assert {"exit_reason", "trade_count", "win_rate", "total_realized_pnl"}.issubset(
        report.exit_attribution.columns
    )


def test_current_strategy_model_summary_uses_fixed_round_manifests():
    models = current_strategy_model_summary()

    assert {"fold_id", "model_role", "model_id", "n_estimators", "objective", "train_rows"}.issubset(
        models.columns
    )
    assert set(models["model_role"]) == {"absolute", "risk"}
    assert models.loc[models["model_role"] == "absolute", "n_estimators"].dropna().eq(160).all()
    assert models.loc[models["model_role"] == "risk", "n_estimators"].dropna().eq(120).all()
    assert models["train_window_mode"].eq("fixed_round_full_history").all()
    assert models["feature_set_id"].eq("vpa_d_sequence_fundamental_v1").all()


def test_current_strategy_report_paths_are_explicit_and_existing():
    paths = strategy_report_paths()

    assert paths["report_dir"].name == "profit_protect_continuous_wf_2020_2025_20260622"
    for path in paths.values():
        assert path.exists(), path
