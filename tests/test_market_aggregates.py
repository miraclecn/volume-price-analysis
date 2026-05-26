import pandas as pd

from vpa_structure_recognizer.market_aggregates import (
    build_market_bars,
    build_sector_bars,
)
from vpa_structure_recognizer.models import MARKET_BAR_COLUMNS, SECTOR_BAR_COLUMNS


def _stock_bars() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "code": "000001.SZ",
                "open": 10.0,
                "high": 11.0,
                "low": 9.8,
                "close": 10.5,
                "prev_close": 10.0,
                "volume": 1000.0,
                "amount": 10500.0,
                "is_st": False,
                "is_paused": False,
                "limit_up": 11.0,
                "limit_down": 9.0,
                "industry_code": "BK001",
                "industry_name": "Banking",
            },
            {
                "date": "2024-01-02",
                "code": "000002.SZ",
                "open": 20.0,
                "high": 20.5,
                "low": 18.0,
                "close": 18.0,
                "prev_close": 20.0,
                "volume": 2000.0,
                "amount": 36000.0,
                "is_st": False,
                "is_paused": False,
                "limit_up": 22.0,
                "limit_down": 18.0,
                "industry_code": "BK002",
                "industry_name": "Real Estate",
            },
            {
                "date": "2024-01-03",
                "code": "000001.SZ",
                "open": 10.5,
                "high": 11.5,
                "low": 10.4,
                "close": 11.2,
                "prev_close": 10.5,
                "volume": 1500.0,
                "amount": 16800.0,
                "is_st": False,
                "is_paused": False,
                "limit_up": 11.55,
                "limit_down": 9.45,
                "industry_code": "BK001",
                "industry_name": "Banking",
            },
        ]
    )


def test_build_market_bars_computes_breadth_and_totals():
    market = build_market_bars(_stock_bars())

    assert list(market.columns) == MARKET_BAR_COLUMNS
    first = market[market["date"] == "2024-01-02"].iloc[0]
    assert first["advancers_count"] == 1
    assert first["decliners_count"] == 1
    assert first["limit_up_count"] == 0
    assert first["limit_down_count"] == 1
    assert first["total_volume"] == 3000.0
    assert first["total_amount"] == 46500.0
    assert first["median_ret_pct"] == -0.025


def test_build_sector_bars_groups_by_industry_code():
    sectors = build_sector_bars(_stock_bars())

    assert list(sectors.columns) == SECTOR_BAR_COLUMNS
    first = sectors[
        (sectors["date"] == "2024-01-02") & (sectors["sector_code"] == "BK001")
    ].iloc[0]
    assert first["sector_name"] == "Banking"
    assert first["advancers_count"] == 1
    assert first["decliners_count"] == 0
    assert first["member_count"] == 1
    assert first["volume"] == 1000.0
