import pandas as pd

from vpa_structure_recognizer.feature_engineering import compute_features


def test_compute_features_uses_percentage_price_metrics_and_window_volume():
    bars = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "code": "000001.SZ",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "prev_close": 10.0,
                "volume": 100.0,
                "amount": 1050.0,
            },
            {
                "date": "2024-01-03",
                "code": "000001.SZ",
                "open": 10.5,
                "high": 12.0,
                "low": 10.0,
                "close": 11.5,
                "prev_close": 10.5,
                "volume": 300.0,
                "amount": 3450.0,
            },
        ]
    )

    features = compute_features(bars, windows=[2], scope_type="stock", scope_id_column="code")
    last = features[features["date"] == "2024-01-03"].iloc[0]

    assert last["ret_pct"] == 11.5 / 10.5 - 1
    assert last["range_pct"] == (12.0 - 10.0) / 10.5
    assert last["body_pct"] == abs(11.5 - 10.5) / 10.5
    assert last["upper_shadow_pct"] == (12.0 - 11.5) / 10.5
    assert last["lower_shadow_pct"] == (10.5 - 10.0) / 10.5
    assert last["body_ratio"] == 0.5
    assert last["close_position"] == 0.75
    assert last["vol_ma_n"] == 200.0
    assert last["vol_rvol_n"] == 1.5
    assert last["price_high_n"] == 12.0
    assert last["price_low_n"] == 9.0
    assert last["prev_price_high_n"] == 11.0
    assert last["prev_price_low_n"] == 9.0
    assert last["price_position_n"] == (11.5 - 9.0) / (12.0 - 9.0)


def test_compute_features_emits_one_row_per_window_per_date():
    bars = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "code": "000001.SZ",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "prev_close": 10.0,
                "volume": 100.0,
                "amount": 1000.0,
            }
        ]
    )

    features = compute_features(bars, windows=[10, 20], scope_type="stock", scope_id_column="code")

    assert features[["date", "scope_type", "scope_id", "window_n"]].to_dict("records") == [
        {
            "date": "2024-01-02",
            "scope_type": "stock",
            "scope_id": "000001.SZ",
            "window_n": 10,
        },
        {
            "date": "2024-01-02",
            "scope_type": "stock",
            "scope_id": "000001.SZ",
            "window_n": 20,
        },
    ]
    assert features["body_ratio"].tolist() == [0.0, 0.0]
    assert features["close_position"].isna().all()
