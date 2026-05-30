from __future__ import annotations

import textwrap

import pytest

from ml_stock_selector.config import load_ml_config
from ml_stock_selector.constants import DEFAULT_FEATURE_WINDOWS, FEATURE_SET_VPA_D


def test_default_ml_config_loads():
    config = load_ml_config("config/ml_default.toml")

    assert config.features["windows"] == DEFAULT_FEATURE_WINDOWS
    assert config.features["feature_set_id"] == FEATURE_SET_VPA_D
    assert config.backtest["execution_price"] == "next_open"


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

