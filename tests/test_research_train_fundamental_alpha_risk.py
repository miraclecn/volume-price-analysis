from __future__ import annotations

import pytest
import pandas as pd

from ml_stock_selector.config import MLConfig
from scripts.research_train_fundamental_alpha_risk import _monotonicity_summary
from scripts.research_train_fundamental_alpha_risk import train_window_for_mode
from scripts.research_train_fundamental_alpha_risk import apply_training_overrides


def test_train_window_for_expanding_no_gap_includes_validation_year():
    manifest = {
        "train_start": "2015-01-05",
        "train_end": "2018-12-31",
        "valid_start": "2019-01-01",
        "valid_end": "2019-12-31",
    }

    assert train_window_for_mode(manifest, "expanding_no_gap") == ("2015-01-05", "2019-12-31")


def test_train_window_for_rolling_modes_uses_last_n_years_through_validation_year():
    manifest = {
        "train_start": "2015-01-05",
        "train_end": "2021-12-31",
        "valid_start": "2022-01-01",
        "valid_end": "2022-12-31",
    }

    assert train_window_for_mode(manifest, "rolling_5y_no_gap") == ("2018-01-01", "2022-12-31")
    assert train_window_for_mode(manifest, "rolling_3y_no_gap") == ("2020-01-01", "2022-12-31")


def test_train_window_for_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown train_window_mode"):
        train_window_for_mode({}, "bad")


def test_apply_training_overrides_updates_lightgbm_runtime_without_mutating_original():
    config = MLConfig(
        data={},
        features={},
        labels={},
        split={},
        universe={},
        model={"lightgbm_runtime": {"n_estimators": 25, "early_stopping_rounds": 0}},
        portfolio={},
        backtest={},
        ml_v2={},
    )

    updated = apply_training_overrides(config, n_estimators=300, early_stopping_rounds=50)

    assert config.model["lightgbm_runtime"]["n_estimators"] == 25
    assert updated.model["lightgbm_runtime"]["n_estimators"] == 300
    assert updated.model["lightgbm_runtime"]["early_stopping_rounds"] == 50


def test_monotonicity_summary_counts_decile_violations():
    deciles = pd.DataFrame(
        {
            "year": [2024, 2024, 2024],
            "score_decile": [1, 2, 3],
            "row_count": [10, 10, 10],
            "mean_alpha_rank_pct": [0.1, 0.2, 0.3],
            "mean_future_max_gain": [0.01, 0.03, 0.02],
            "mean_future_ret": [0.00, 0.01, 0.02],
            "positive_ret_rate": [0.4, 0.5, 0.6],
        }
    )

    summary = _monotonicity_summary(deciles)
    row = summary[summary["year"].eq(2024)].iloc[0]

    assert row["future_max_gain_violation_count"] == 1
    assert bool(row["future_max_gain_monotonic"]) is False
    assert row["future_ret_violation_count"] == 0
    assert bool(row["future_ret_monotonic"]) is True
