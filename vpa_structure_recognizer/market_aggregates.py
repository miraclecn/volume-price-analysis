from __future__ import annotations

import pandas as pd

from vpa_structure_recognizer.models import MARKET_BAR_COLUMNS, SECTOR_BAR_COLUMNS


def build_market_bars(stock_bars: pd.DataFrame) -> pd.DataFrame:
    bars = _with_returns(stock_bars)
    bars = _with_new_high_low_flags(bars, [20, 60])

    rows: list[dict[str, object]] = []
    for date, group in bars.groupby("date", sort=True):
        rows.append(
            {
                "date": date,
                "all_a_equal_weight_open": group["open"].mean(),
                "all_a_equal_weight_high": group["high"].mean(),
                "all_a_equal_weight_low": group["low"].mean(),
                "all_a_equal_weight_close": group["close"].mean(),
                "total_amount": group["amount"].sum(),
                "total_volume": group["volume"].sum(),
                "advancers_count": int((group["ret_pct"] > 0).sum()),
                "decliners_count": int((group["ret_pct"] < 0).sum()),
                "limit_up_count": _count_limit_up(group),
                "limit_down_count": _count_limit_down(group),
                "new_high_count_20": int(group["new_high_20"].sum()),
                "new_low_count_20": int(group["new_low_20"].sum()),
                "new_high_count_60": int(group["new_high_60"].sum()),
                "new_low_count_60": int(group["new_low_60"].sum()),
                "strong_stock_ratio": round(float((group["ret_pct"] > 0.03).mean()), 12),
                "weak_stock_ratio": round(float((group["ret_pct"] < -0.03).mean()), 12),
                "median_ret_pct": round(float(group["ret_pct"].median()), 12),
            }
        )

    return pd.DataFrame(rows, columns=MARKET_BAR_COLUMNS)


def build_sector_bars(stock_bars: pd.DataFrame) -> pd.DataFrame:
    bars = _with_returns(stock_bars)
    rows: list[dict[str, object]] = []
    group_columns = ["date", "industry_code", "industry_name"]
    for (date, code, name), group in bars.groupby(group_columns, sort=True, dropna=False):
        rows.append(
            {
                "date": date,
                "sector_code": code,
                "sector_name": name,
                "open": group["open"].mean(),
                "high": group["high"].mean(),
                "low": group["low"].mean(),
                "close": group["close"].mean(),
                "prev_close": group["prev_close"].mean(),
                "volume": group["volume"].sum(),
                "amount": group["amount"].sum(),
                "advancers_count": int((group["ret_pct"] > 0).sum()),
                "decliners_count": int((group["ret_pct"] < 0).sum()),
                "limit_up_count": _count_limit_up(group),
                "limit_down_count": _count_limit_down(group),
                "member_count": int(group["code"].nunique()),
            }
        )

    return pd.DataFrame(rows, columns=SECTOR_BAR_COLUMNS)


def _with_returns(stock_bars: pd.DataFrame) -> pd.DataFrame:
    bars = stock_bars.copy()
    bars["ret_pct"] = bars["close"] / bars["prev_close"] - 1
    return bars


def _with_new_high_low_flags(bars: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    output = bars.sort_values(["code", "date"]).copy()
    grouped = output.groupby("code", sort=False)
    for window in windows:
        rolling_high = grouped["close"].transform(
            lambda values: values.rolling(window, min_periods=1).max()
        )
        rolling_low = grouped["close"].transform(
            lambda values: values.rolling(window, min_periods=1).min()
        )
        output[f"new_high_{window}"] = output["close"] >= rolling_high
        output[f"new_low_{window}"] = output["close"] <= rolling_low
    return output


def _count_limit_up(group: pd.DataFrame) -> int:
    comparable = group["limit_up"].notna()
    return int((comparable & (group["close"] >= group["limit_up"])).sum())


def _count_limit_down(group: pd.DataFrame) -> int:
    comparable = group["limit_down"].notna()
    return int((comparable & (group["close"] <= group["limit_down"])).sum())
