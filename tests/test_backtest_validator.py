import pandas as pd

from vpa_structure_recognizer.backtest_validator import compute_validation_metrics


def test_compute_validation_metrics_adds_future_return_and_risk_fields():
    states = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "scope_type": "stock",
                "scope_id": "000001.SZ",
                "sector_id": "BK001",
                "final_state": "HEALTHY_UPTREND",
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {"date": f"2024-01-{day:02d}", "code": "000001.SZ", "close": close, "high": high, "low": low}
            for day, close, high, low in [
                (1, 10.0, 10.2, 9.8),
                (2, 10.5, 10.8, 10.1),
                (3, 11.0, 11.2, 10.7),
                (4, 9.5, 10.0, 9.0),
                (5, 12.0, 12.5, 11.5),
                (6, 13.0, 13.2, 12.8),
            ]
        ]
    )
    sector_returns = pd.DataFrame(
        [{"date": "2024-01-01", "sector_id": "BK001", "future_ret_10d": 0.1}]
    )
    market_returns = pd.DataFrame([{"date": "2024-01-01", "future_ret_10d": 0.05}])

    result = compute_validation_metrics(states, prices, sector_returns, market_returns)
    row = result.iloc[0]

    assert row["future_ret_1d"] == 0.05
    assert row["future_ret_3d"] == -0.05
    assert row["future_ret_5d"] == 0.3
    assert row["future_max_gain_10d"] == 0.32
    assert row["future_max_drawdown_10d"] == -0.1
    assert row["hit_new_high_20d"] is True
    assert row["hit_new_low_20d"] is True
    assert row["outperform_sector_10d"] is True
    assert row["outperform_market_10d"] is True


def test_validation_metrics_leave_missing_future_horizon_empty():
    states = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "scope_type": "stock",
                "scope_id": "000001.SZ",
                "final_state": "LOW_LEVEL_SUPPORT",
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "code": "000001.SZ", "close": 10.0, "high": 10.0, "low": 10.0},
            {"date": "2024-01-02", "code": "000001.SZ", "close": 10.1, "high": 10.2, "low": 10.0},
        ]
    )

    result = compute_validation_metrics(states, prices)

    assert result.iloc[0]["future_ret_1d"] == 0.01
    assert result.iloc[0]["future_ret_3d"] is None
