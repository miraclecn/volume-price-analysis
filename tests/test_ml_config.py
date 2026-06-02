from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ml_stock_selector.config import load_ml_config
from ml_stock_selector.constants import DEFAULT_FEATURE_WINDOWS, FEATURE_SET_VPA_D


def test_default_ml_config_loads():
    config = load_ml_config("config/ml_default.toml")

    assert config.features["windows"] == DEFAULT_FEATURE_WINDOWS
    assert config.features["feature_set_id"] == FEATURE_SET_VPA_D
    assert config.backtest["execution_price"] == "next_open"


def test_default_ml_config_exposes_v2_flags_enabled():
    config = load_ml_config("config/ml_default.toml")

    assert config.ml_v2["exclude_industry_metadata_from_features_json"] is True
    assert config.ml_v2["feature_matrix_v2_deny_industry"] is True
    assert config.ml_v2["labels_v2_enabled"] is True
    assert config.ml_v2["active_ranker_enabled"] is True
    assert config.ml_v2["risk_model_v2_enabled"] is True
    assert config.ml_v2["trade_score_v2_enabled"] is True
    assert config.ml_v2["daily_signal_v2_enabled"] is True
    assert config.universe["exclude_bse"] is True
    assert config.portfolio["v2"]["holding"] == {
        "min_hold_days": 3,
        "target_hold_days": 5,
        "max_hold_days": 10,
    }
    assert config.portfolio["v2"]["exit"]["sell_score_threshold"] == 0.45
    assert config.portfolio["v2"]["exit"]["sell_score_threshold"] < config.portfolio["v2"]["candidate_min_trade_score"]


def test_walkforward_config_uses_2015_start_and_named_folds():
    config = load_ml_config("config/ml_walkforward.toml")
    folds = config.split["folds"]
    assert folds
    assert all(fold["train_start"] == "2015-01-05" for fold in folds)
    ids = [fold["fold_id"] for fold in folds]
    assert len(ids) == len(set(ids))


def test_ml_v2_config_overrides_are_loaded(tmp_path):
    path = tmp_path / "ml-v2.toml"
    path.write_text(
        textwrap.dedent(
            """
            [data]
            alpha_data_db = "a.duckdb"
            vpa_db = "v.duckdb"
            ml_db = "m.duckdb"
            normalized_bars_table = "stock_bar_normalized_daily"
            artifact_dir = "outputs/ml/artifacts"
            report_dir = "outputs/ml/reports"

            [features]
            windows = [5, 10]
            feature_set_id = "vpa_d_sequence"
            include_structure_state = false

            [labels]
            horizons = [5, 10]
            main_horizon = 10
            label_base = "from_next_open"
            risk_drawdown_threshold = -0.05

            [split]
            embargo_days = 10
            folds = []

            [model.alpha_ranker]
            objective = "lambdarank"

            [portfolio]
            target_positions = 12
            hard_max_positions = 15
            max_industry_names = 3
            max_new_entries_per_day = 4
            single_name_min_weight = 0.05
            single_name_max_weight = 0.10
            allow_cash = true
            min_trade_score = 0.80

            [backtest]
            initial_cash = 1000000
            execution_price = "next_open"
            slippage_bps = 5
            commission_bps = 3
            stamp_duty_bps = 5
            a_share_lot_size = 100
            allow_fractional_shares = true

            [ml_v2]
            exclude_industry_metadata_from_features_json = true
            feature_matrix_v2_deny_industry = true
            labels_v2_enabled = true
            active_ranker_enabled = true
            risk_model_v2_enabled = true
            trade_score_v2_enabled = true
            daily_signal_v2_enabled = true
            candidate_absolute_min_rank_pct = 0.70
            candidate_active_min_rank_pct = 0.71
            candidate_risk_max_rank_pct = 0.60
            core_absolute_min_rank_pct = 0.80
            core_active_min_rank_pct = 0.75
            core_risk_max_rank_pct = 0.35
            core_min_trade_score = 0.82
            """
        ),
        encoding="utf-8",
    )

    config = load_ml_config(path)

    assert config.ml_v2["exclude_industry_metadata_from_features_json"] is True
    assert config.ml_v2["feature_matrix_v2_deny_industry"] is True
    assert config.ml_v2["labels_v2_enabled"] is True
    assert config.ml_v2["active_ranker_enabled"] is True
    assert config.ml_v2["risk_model_v2_enabled"] is True
    assert config.ml_v2["trade_score_v2_enabled"] is True
    assert config.ml_v2["daily_signal_v2_enabled"] is True
    assert config.ml_v2["candidate_active_min_rank_pct"] == 0.71
    assert config.ml_v2["core_min_trade_score"] == 0.82


def test_invalid_weight_bounds_are_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        textwrap.dedent(
            """
            [data]
            alpha_data_db = "a.duckdb"
            vpa_db = "v.duckdb"
            ml_db = "m.duckdb"
            normalized_bars_table = "stock_bar_normalized_daily"
            artifact_dir = "outputs/ml/artifacts"
            report_dir = "outputs/ml/reports"

            [features]
            windows = [5]
            feature_set_id = "vpa_d_sequence"
            include_structure_state = false

            [labels]
            horizons = [5]
            main_horizon = 5
            label_base = "from_next_open"
            risk_drawdown_threshold = -0.05

            [split]
            embargo_days = 10
            folds = []

            [model.alpha_ranker]
            objective = "lambdarank"

            [portfolio]
            target_positions = 12
            hard_max_positions = 15
            max_industry_names = 3
            max_new_entries_per_day = 4
            single_name_min_weight = 0.11
            single_name_max_weight = 0.10
            allow_cash = true
            min_trade_score = 0.80

            [backtest]
            initial_cash = 1000000
            execution_price = "next_open"
            slippage_bps = 5
            commission_bps = 3
            stamp_duty_bps = 5
            a_share_lot_size = 100
            allow_fractional_shares = true
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="single_name_min_weight"):
        load_ml_config(path)


def test_invalid_sell_threshold_without_hysteresis_is_rejected(tmp_path):
    path = tmp_path / "bad-sell-threshold.toml"
    text = Path("config/ml_default.toml").read_text(encoding="utf-8")
    path.write_text(text.replace("sell_score_threshold = 0.45", "sell_score_threshold = 0.65"), encoding="utf-8")

    with pytest.raises(ValueError, match="sell_score_threshold"):
        load_ml_config(path)
