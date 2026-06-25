from __future__ import annotations

import pandas as pd
import pytest

from ml_stock_selector.tradeability import build_tradeability_mart
from tests.ml_fixtures import normalized_bars


def test_tradeability_uses_next_open_and_rejects_paused_or_limits():
    bars = normalized_bars()
    bars.loc[(bars["code"] == "000001.SZ") & (bars["trade_date"] == "2024-01-03"), "open"] = 11.0
    bars.loc[(bars["code"] == "000001.SZ") & (bars["trade_date"] == "2024-01-03"), "limit_up"] = 11.0
    mart = build_tradeability_mart(bars, adv_window=2)

    row = mart[(mart["code"] == "000001.SZ") & (mart["trade_date"] == "2024-01-02")].iloc[0]
    assert row["next_trade_date"] == "2024-01-03"
    assert row["can_buy_next_open"] is False

    last = mart[(mart["code"] == "000001.SZ") & (mart["trade_date"] == "2024-01-08")].iloc[0]
    assert last["can_buy_next_open"] is False
    assert last["can_sell_next_open"] is False


def test_tradeability_derives_limit_band_from_source_limit_prices():
    mart = build_tradeability_mart(_limit_band_bars())

    ten = mart[mart["code"] == "000001.SZ"].iloc[0]
    twenty = mart[mart["code"] == "300001.SZ"].iloc[0]

    assert ten["limit_up_pct"] == pytest.approx(0.10)
    assert ten["limit_down_pct"] == pytest.approx(-0.10)
    assert ten["limit_band"] == "limit_10pct"
    assert twenty["limit_up_pct"] == pytest.approx(0.20)
    assert twenty["limit_down_pct"] == pytest.approx(-0.20)
    assert twenty["limit_band"] == "limit_20pct"


def _limit_band_bars() -> pd.DataFrame:
    rows = []
    for code, limit_up, limit_down in [
        ("000001.SZ", 11.0, 9.0),
        ("300001.SZ", 12.0, 8.0),
    ]:
        rows.append(
            {
                "trade_date": "2024-01-02",
                "code": code,
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.0,
                "prev_close": 10.0,
                "volume": 1000,
                "amount": 10000.0,
                "turnover_rate": 1.0,
                "is_st": False,
                "is_paused": False,
                "limit_up": limit_up,
                "limit_down": limit_down,
                "industry_code": "I1",
                "industry_name": "Industry",
            }
        )
    return pd.DataFrame(rows)
