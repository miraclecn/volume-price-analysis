from __future__ import annotations

import pandas as pd
import pytest

from ml_stock_selector.label_builder import build_labels, rank_label_from_pct
from tests.ml_fixtures import normalized_bars


def test_labels_are_long_format_and_use_configured_bases():
    labels = build_labels(normalized_bars(), [1, 2], label_bases=["from_close", "from_next_open"])

    assert {"from_close", "from_next_open"} == set(labels["label_base"])
    assert {"trade_date", "code", "horizon_d", "future_ret", "rank_label"}.issubset(labels.columns)
    assert labels[["trade_date", "code", "horizon_d", "label_base"]].duplicated().sum() == 0


def test_rank_label_is_head_heavy():
    assert rank_label_from_pct(0.995) == 4
    assert rank_label_from_pct(0.96) == 3
    assert rank_label_from_pct(0.91) == 2
    assert rank_label_from_pct(0.75) == 1
    assert rank_label_from_pct(0.30) == 0


def test_v2_labels_add_absolute_and_active_return_fields():
    labels = build_labels(normalized_bars(), [1], label_bases=["from_close"], include_v2=True)

    assert {
        "absolute_ret",
        "absolute_rank_pct",
        "absolute_label",
        "market_ret",
        "industry_ret",
        "market_excess_ret",
        "industry_excess_ret",
        "active_score",
        "active_rank_pct",
        "active_label",
        "benchmark_missing_market",
        "benchmark_missing_industry",
        "benchmark_peer_count",
    }.issubset(labels.columns)
    assert (labels["absolute_ret"] == labels["future_ret"]).all()
    assert (labels["absolute_label"] == labels["rank_label"]).all()
    assert labels["active_label"].notna().all()


def test_v2_market_excess_uses_cross_sectional_benchmark():
    bars = _benchmark_fixture()

    labels = build_labels(bars, [1], label_bases=["from_close"], include_v2=True)
    same_market = labels[(labels["trade_date"] == "2024-01-02") & (labels["code"] == "A")].iloc[0]
    down_market = labels[(labels["trade_date"] == "2024-01-03") & (labels["code"] == "B")].iloc[0]

    assert same_market["market_ret"] == pytest.approx(0.05)
    assert same_market["market_excess_ret"] == pytest.approx(0.0)
    assert down_market["market_ret"] == pytest.approx(-0.03)
    assert down_market["market_excess_ret"] == pytest.approx(0.04)


def test_v2_unknown_industry_falls_back_to_market_excess_for_active_score():
    bars = _benchmark_fixture()
    bars.loc[bars["code"] == "B", "industry_code"] = "UNKNOWN"
    bars.loc[bars["code"] == "B", "industry_name"] = "UNKNOWN"

    labels = build_labels(bars, [1], label_bases=["from_close"], include_v2=True)
    row = labels[(labels["trade_date"] == "2024-01-03") & (labels["code"] == "B")].iloc[0]

    assert row["benchmark_missing_industry"] is True
    assert pd.isna(row["industry_ret"])
    assert row["active_score"] == pytest.approx(row["market_excess_ret"])


def _benchmark_fixture() -> pd.DataFrame:
    rows = []
    closes = {
        "A": [100.0, 105.0, 97.65],
        "B": [100.0, 105.0, 106.05],
    }
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    for code, values in closes.items():
        for date, close in zip(dates, values, strict=True):
            rows.append(
                {
                    "trade_date": date,
                    "code": code,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "prev_close": close,
                    "volume": 1000,
                    "amount": 1000 * close,
                    "turnover_rate": 1.0,
                    "is_st": False,
                    "is_paused": False,
                    "limit_up": close * 1.1,
                    "limit_down": close * 0.9,
                    "industry_code": "I1",
                    "industry_name": "Industry 1",
                }
            )
    return pd.DataFrame(rows)
