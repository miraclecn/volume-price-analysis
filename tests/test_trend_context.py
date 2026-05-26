import pandas as pd

from vpa_structure_recognizer.trend_context import compute_trend_context


def test_trend_context_uses_parent_window_metrics():
    features = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "scope_type": "stock",
                "scope_id": "000001.SZ",
                "window_n": 20,
                "close": 80.0,
                "price_high_n": 85.0,
                "price_low_n": 75.0,
                "ma_n": 79.0,
                "ma_slope_n": 0.0,
            },
            {
                "date": "2024-01-02",
                "scope_type": "stock",
                "scope_id": "000001.SZ",
                "window_n": 60,
                "close": 80.0,
                "price_high_n": 100.0,
                "price_low_n": 0.0,
                "ma_n": 50.0,
                "ma_slope_n": 0.02,
            },
        ]
    )

    context = compute_trend_context(features, {20: [60], 60: [240]})
    row = context[context["window_n"] == 20].iloc[0]

    assert row["parent_window_n"] == 60
    assert row["parent_high"] == 100.0
    assert row["parent_low"] == 0.0
    assert row["parent_price_position"] == 0.8
    assert row["trend_label"] == "UPTREND"
    assert row["position_label"] == "MID_HIGH"


def test_position_label_boundaries_are_spec_aligned():
    features = pd.DataFrame(
        [
            {
                "date": f"2024-01-0{idx + 1}",
                "scope_type": "stock",
                "scope_id": "000001.SZ",
                "window_n": 10,
                "close": close,
                "price_high_n": close,
                "price_low_n": close,
                "ma_n": close,
                "ma_slope_n": 0.0,
            }
            for idx, close in enumerate([24.0, 25.0, 45.0, 65.0, 85.0])
        ]
        + [
            {
                "date": f"2024-01-0{idx + 1}",
                "scope_type": "stock",
                "scope_id": "000001.SZ",
                "window_n": 30,
                "close": close,
                "price_high_n": 100.0,
                "price_low_n": 0.0,
                "ma_n": 50.0,
                "ma_slope_n": 0.0,
            }
            for idx, close in enumerate([24.0, 25.0, 45.0, 65.0, 85.0])
        ]
    )

    context = compute_trend_context(features, {10: [30]})
    labels = context[context["window_n"] == 10]["position_label"].tolist()

    assert labels == ["LOW", "MID_LOW", "MID", "MID_HIGH", "HIGH"]


def test_missing_parent_window_returns_unknown_context():
    features = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "scope_type": "stock",
                "scope_id": "000001.SZ",
                "window_n": 10,
                "close": 10.0,
                "price_high_n": 10.0,
                "price_low_n": 10.0,
                "ma_n": 10.0,
                "ma_slope_n": 0.0,
            }
        ]
    )

    context = compute_trend_context(features, {10: [30]})
    row = context.iloc[0]

    assert row["trend_label"] == "UNKNOWN"
    assert row["position_label"] == "UNKNOWN"
