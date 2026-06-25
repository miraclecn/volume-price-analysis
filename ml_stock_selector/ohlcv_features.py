from __future__ import annotations

import pandas as pd


def build_ohlcv_features(normalized_bars: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    rows = []
    for code, group in normalized_bars.sort_values(["code", "trade_date"]).groupby("code", sort=False):
        g = group.reset_index(drop=True).copy()
        prev_close = g["close"].shift(1)
        g["ret_1d"] = g["close"] / prev_close - 1.0
        g["open_gap_pct"] = g["open"] / prev_close - 1.0
        g["range_pct"] = (g["high"] - g["low"]) / g["close"]
        g["body_pct"] = (g["close"] - g["open"]) / g["open"]
        g["upper_shadow_pct"] = (g["high"] - g[["open", "close"]].max(axis=1)) / g["close"]
        g["lower_shadow_pct"] = (g[["open", "close"]].min(axis=1) - g["low"]) / g["open"]
        denom = (g["high"] - g["low"]).replace(0, pd.NA)
        g["close_position"] = (g["close"] - g["low"]) / denom
        for window in windows:
            rolling_amount = g["amount"].rolling(window, min_periods=1).mean()
            rolling_volume = g["volume"].rolling(window, min_periods=1).mean()
            g[f"ret_{window}d"] = g["close"] / g["close"].shift(window) - 1.0
            g[f"volatility_{window}d"] = g["close"].pct_change().rolling(window, min_periods=1).std().fillna(0.0)
            g[f"amount_ratio_{window}d"] = g["amount"] / rolling_amount
            g[f"volume_ratio_{window}d"] = g["volume"] / rolling_volume
            g[f"turnover_mean_{window}d"] = g["turnover_rate"].rolling(window, min_periods=1).mean()
            high_n = g["high"].rolling(window, min_periods=1).max().shift(1)
            low_n = g["low"].rolling(window, min_periods=1).min().shift(1)
            g[f"high_distance_{window}d"] = g["close"] / high_n - 1.0
            g[f"low_distance_{window}d"] = g["close"] / low_n - 1.0
        g["code"] = code
        rows.append(g)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out.astype(object).where(pd.notna(out), None)
