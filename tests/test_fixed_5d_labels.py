from __future__ import annotations

import pandas as pd
import pytest

from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.sample_builder import build_training_samples


def test_fixed_5d_absolute_label_uses_next_open_to_fifth_future_close():
    bars = pd.DataFrame(
        [
            {"trade_date": f"2024-01-0{i + 1}", "code": "000001.SZ", "open": 10 + i, "high": 10 + i, "low": 10 + i, "close": 10 + i, "industry_code": "I1"}
            for i in range(7)
        ]
    )

    labels = build_labels(bars, [5], label_bases=["from_next_open"], include_v2=True)
    row = labels[(labels["trade_date"] == "2024-01-01") & (labels["horizon_d"] == 5)].iloc[0]

    assert row["label_base"] == "from_next_open"
    assert row["base_price"] == 11.0
    assert row["absolute_ret"] == pytest.approx(15.0 / 11.0 - 1.0)


def test_executable_only_training_samples_filter_tradeability_without_feature_leakage():
    feature_mart = pd.DataFrame(
        {
            "trade_date": ["2024-01-01"] * 5,
            "code": ["good", "bse", "st", "paused", "illiquid"],
            "feature_set_id": ["set"] * 5,
            "features_json": ['{"x": 1.0}'] * 5,
            "can_buy_next_open": [False, True, True, True, True],
            "is_bse": [False, True, False, False, False],
            "is_st": [False, False, True, False, False],
            "is_paused": [False, False, False, True, False],
            "adv20_amount": [100_000_000, 100_000_000, 100_000_000, 100_000_000, 1_000],
        }
    )
    labels = pd.DataFrame(
        {
            "trade_date": ["2024-01-01"] * 5,
            "code": ["good", "bse", "st", "paused", "illiquid"],
            "horizon_d": [5] * 5,
            "label_base": ["from_next_open"] * 5,
            "absolute_label": [1] * 5,
            "absolute_ret": [0.1] * 5,
            "risk_label": [0] * 5,
        }
    )

    samples = build_training_samples(
        feature_mart,
        labels,
        "set",
        5,
        "from_next_open",
        label_name="absolute_label",
        executable_only=True,
        min_adv20_amount=50_000_000,
    )

    assert samples["code"].tolist() == ["good"]
