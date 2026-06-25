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


def test_rank_label_accepts_configured_thresholds():
    thresholds = [
        {"label": 2, "min_rank_pct": 0.80},
        {"label": 1, "min_rank_pct": 0.50},
    ]

    assert rank_label_from_pct(0.85, thresholds) == 2
    assert rank_label_from_pct(0.60, thresholds) == 1
    assert rank_label_from_pct(0.40, thresholds) == 0


def test_labels_accept_configured_future_score_weights():
    labels = build_labels(
        _score_weight_fixture(),
        [1],
        label_bases=["from_close"],
        future_score_weights={
            "future_ret": 2.0,
            "future_max_gain": 0.0,
            "future_max_drawdown_abs": 0.0,
        },
    )

    row = labels[(labels["trade_date"] == "2024-01-02") & (labels["code"] == "A")].iloc[0]
    assert row["future_ret"] == pytest.approx(0.10)
    assert row["future_score"] == pytest.approx(0.20)


def test_labels_can_rank_within_limit_band():
    labels = build_labels(
        _limit_band_rank_fixture(),
        [1],
        label_bases=["from_close"],
        rank_group_by_limit_band=True,
    )

    ten_top = labels[(labels["trade_date"] == "2024-01-02") & (labels["code"] == "B")].iloc[0]
    twenty_top = labels[(labels["trade_date"] == "2024-01-02") & (labels["code"] == "D")].iloc[0]
    assert ten_top["limit_band"] == "limit_10pct"
    assert twenty_top["limit_band"] == "limit_20pct"
    assert ten_top["future_rank_pct"] == pytest.approx(1.0)
    assert twenty_top["future_rank_pct"] == pytest.approx(1.0)
    assert ten_top["rank_label"] == 4
    assert twenty_top["rank_label"] == 4


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


def test_labels_keep_signal_day_limit_band_metadata():
    bars = _benchmark_fixture()
    bars.loc[bars["code"] == "A", ["prev_close", "limit_up", "limit_down"]] = [100.0, 110.0, 90.0]
    bars.loc[bars["code"] == "B", ["prev_close", "limit_up", "limit_down"]] = [100.0, 120.0, 80.0]

    labels = build_labels(bars, [1], label_bases=["from_close"])

    ten = labels[(labels["trade_date"] == "2024-01-02") & (labels["code"] == "A")].iloc[0]
    twenty = labels[(labels["trade_date"] == "2024-01-02") & (labels["code"] == "B")].iloc[0]
    assert ten["limit_up_pct"] == pytest.approx(0.10)
    assert ten["limit_down_pct"] == pytest.approx(-0.10)
    assert ten["limit_band"] == "limit_10pct"
    assert twenty["limit_up_pct"] == pytest.approx(0.20)
    assert twenty["limit_down_pct"] == pytest.approx(-0.20)
    assert twenty["limit_band"] == "limit_20pct"


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


def _score_weight_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _bar("2024-01-02", "A", 100.0, 100.0, 110.0, 90.0),
            _bar("2024-01-03", "A", 110.0, 110.0, 121.0, 99.0),
        ]
    )


def _limit_band_rank_fixture() -> pd.DataFrame:
    rows = []
    specs = [
        ("A", 100.0, 101.0, 0.10),
        ("B", 100.0, 102.0, 0.10),
        ("C", 100.0, 103.0, 0.20),
        ("D", 100.0, 104.0, 0.20),
    ]
    for code, start, future, limit_width in specs:
        rows.append(_bar("2024-01-02", code, start, start, start * (1 + limit_width), start * (1 - limit_width)))
        rows.append(_bar("2024-01-03", code, future, future, future * (1 + limit_width), future * (1 - limit_width)))
    return pd.DataFrame(rows)


def _bar(
    trade_date: str,
    code: str,
    close: float,
    prev_close: float,
    limit_up: float,
    limit_down: float,
) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "code": code,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "prev_close": prev_close,
        "volume": 1000,
        "amount": 1000 * close,
        "turnover_rate": 1.0,
        "is_st": False,
        "is_paused": False,
        "limit_up": limit_up,
        "limit_down": limit_down,
        "industry_code": "I1",
        "industry_name": "Industry 1",
    }
