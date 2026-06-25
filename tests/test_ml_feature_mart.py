from __future__ import annotations

import json

import pytest

from ml_stock_selector.constants import (
    FEATURE_SET_BASELINE_A,
    FEATURE_SET_VPA_D,
    FEATURE_SET_VPA_E,
)
from ml_stock_selector.feature_mart import (
    apply_feature_set_filter,
    build_feature_mart,
    build_structure_state_features,
    build_vpa_bar_context_features,
    build_vpa_numeric_features,
    build_vpa_sequence_features,
)
from ml_stock_selector.tradeability import build_tradeability_mart
from tests.ml_fixtures import create_vpa_db, normalized_bars


def test_vpa_feature_builders_pivot_by_window(tmp_path):
    vpa_db = create_vpa_db(tmp_path / "vpa.duckdb")

    numeric = build_vpa_numeric_features(str(vpa_db), "2024-01-02", "2024-01-08", [5, 20])
    bars = build_vpa_bar_context_features(str(vpa_db), "2024-01-02", "2024-01-08", [5, 20])
    seq = build_vpa_sequence_features(str(vpa_db), "2024-01-02", "2024-01-08", [5, 20])

    assert {"ret_pct_5", "ret_pct_20"}.issubset(numeric.columns)
    assert "raw_label_20" in bars.columns
    assert "support_label_count_20" in seq.columns
    assert numeric[["trade_date", "code"]].duplicated().sum() == 0


def test_feature_set_filter_keeps_structure_state_only_for_vpa_e(tmp_path):
    vpa_db = create_vpa_db(tmp_path / "vpa.duckdb")
    state = build_structure_state_features(str(vpa_db), "2024-01-02", "2024-01-08")

    assert "final_state" not in apply_feature_set_filter(state, FEATURE_SET_VPA_D).columns
    assert "final_state" in apply_feature_set_filter(state, FEATURE_SET_VPA_E).columns


def test_feature_mart_assembles_features_json_by_feature_set(tmp_path):
    vpa_db = create_vpa_db(tmp_path / "vpa.duckdb")
    bars = normalized_bars()
    tradeability = build_tradeability_mart(bars)

    baseline = build_feature_mart(str(vpa_db), bars, "2024-01-02", "2024-01-08", FEATURE_SET_BASELINE_A, [5, 20], tradeability)
    vpa_d = build_feature_mart(str(vpa_db), bars, "2024-01-02", "2024-01-08", FEATURE_SET_VPA_D, [5, 20], tradeability)

    baseline_features = json.loads(baseline.iloc[0]["features_json"])
    vpa_features = json.loads(vpa_d.iloc[0]["features_json"])
    assert "ret_1d" in baseline_features
    assert "raw_label_5" not in baseline_features
    assert "raw_label_5" in vpa_features
    assert "final_state" not in vpa_features


def test_feature_mart_v2_excludes_industry_from_features_json(tmp_path):
    vpa_db = create_vpa_db(tmp_path / "vpa.duckdb")
    bars = normalized_bars()
    tradeability = build_tradeability_mart(bars)

    mart = build_feature_mart(
        str(vpa_db),
        bars,
        "2024-01-02",
        "2024-01-08",
        FEATURE_SET_BASELINE_A,
        [5, 20],
        tradeability,
        exclude_industry_metadata_from_features_json=True,
    )

    features = json.loads(mart.iloc[0]["features_json"])
    assert "industry_code" not in features
    assert "industry_name" not in features
    assert "industry_unknown" not in features
    assert "limit_band" not in features
    assert {"industry_code", "industry_name"}.issubset(mart.columns)
    assert {"limit_up_pct", "limit_down_pct", "limit_band"}.issubset(mart.columns)
    assert mart["industry_code"].notna().any()


def test_feature_mart_uses_warmup_bars_for_ohlcv_features(tmp_path):
    vpa_db = create_vpa_db(tmp_path / "vpa.duckdb")
    bars = normalized_bars()
    tradeability = build_tradeability_mart(bars)

    mart = build_feature_mart(
        str(vpa_db),
        bars,
        "2024-01-03",
        "2024-01-08",
        FEATURE_SET_BASELINE_A,
        [2],
        tradeability,
        exclude_industry_metadata_from_features_json=True,
    )

    row = mart[(mart["code"] == "000001.SZ") & (mart["trade_date"] == "2024-01-03")].iloc[0]
    features = json.loads(row["features_json"])
    assert features["ret_1d"] == pytest.approx(10.2 / 10.0 - 1.0)
    assert features["high_distance_2d"] == pytest.approx(10.2 / 10.2 - 1.0)
