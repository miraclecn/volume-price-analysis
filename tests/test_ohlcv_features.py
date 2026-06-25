from __future__ import annotations

import pytest

from ml_stock_selector.ohlcv_features import build_ohlcv_features
from tests.ml_fixtures import normalized_bars


def test_ohlcv_features_use_only_current_and_prior_bars():
    bars = normalized_bars()
    features = build_ohlcv_features(bars, [2])
    row = features[(features["code"] == "000001.SZ") & (features["trade_date"] == "2024-01-04")].iloc[0]

    assert row["ret_1d"] == pytest.approx(10.4 / 10.2 - 1.0)
    assert row["ret_2d"] == pytest.approx(10.4 / 10.0 - 1.0)
    assert row["open_gap_pct"] == pytest.approx((10.35 / 10.2) - 1.0)
    assert row["close_position"] == pytest.approx((10.4 - 10.15) / (10.6 - 10.15))


def test_ohlcv_high_low_distance_uses_prior_window_extremes():
    bars = normalized_bars()
    features = build_ohlcv_features(bars, [2])
    row = features[(features["code"] == "000001.SZ") & (features["trade_date"] == "2024-01-04")].iloc[0]

    assert row["high_distance_2d"] == pytest.approx(10.4 / 10.4 - 1.0)
    assert row["low_distance_2d"] == pytest.approx(10.4 / 9.75 - 1.0)
