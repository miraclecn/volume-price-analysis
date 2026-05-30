from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def build_tradeability_mart(normalized_bars: pd.DataFrame, adv_window: int = 20) -> pd.DataFrame:
    rows = []
    generated_at = datetime.now(timezone.utc).isoformat()
    for _, group in normalized_bars.sort_values(["code", "trade_date"]).groupby("code", sort=False):
        g = group.reset_index(drop=True).copy()
        g["adv20_amount"] = g["amount"].rolling(adv_window, min_periods=1).mean()
        for column in ["trade_date", "open", "limit_up", "limit_down", "is_paused"]:
            g[f"next_{column}"] = g[column].shift(-1)
        g = g.rename(columns={"next_trade_date": "next_trade_date"})
        g["can_buy_next_open"] = (~g["next_is_paused"].fillna(True).astype(bool)) & (g["next_open"] < g["next_limit_up"])
        g["can_sell_next_open"] = (~g["next_is_paused"].fillna(True).astype(bool)) & (g["next_open"] > g["next_limit_down"])
        g["generated_at"] = generated_at
        rows.append(g)
    columns = [
        "trade_date",
        "code",
        "industry_code",
        "industry_name",
        "is_st",
        "is_paused",
        "limit_up",
        "limit_down",
        "open",
        "high",
        "low",
        "close",
        "prev_close",
        "amount",
        "turnover_rate",
        "adv20_amount",
        "next_trade_date",
        "next_open",
        "next_limit_up",
        "next_limit_down",
        "next_is_paused",
        "can_buy_next_open",
        "can_sell_next_open",
        "generated_at",
    ]
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    return out[columns].astype(object).where(pd.notna(out[columns]), None)

