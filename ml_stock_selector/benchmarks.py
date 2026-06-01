from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def build_benchmark_tables(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if labels.empty:
        return pd.DataFrame(), pd.DataFrame()
    generated_at = datetime.now(timezone.utc).isoformat()
    market_cols = ["trade_date", "horizon_d", "label_base", "market_ret"]
    market_count = (
        labels.groupby(["trade_date", "horizon_d", "label_base"], as_index=False)
        .agg(benchmark_peer_count=("code", "count"))
    )
    market = (
        labels[market_cols]
        .dropna(subset=["market_ret"])
        .drop_duplicates(subset=["trade_date", "horizon_d", "label_base"])
        .copy()
    )
    if not market.empty:
        market = market.merge(market_count, on=["trade_date", "horizon_d", "label_base"], how="left")
        market["generated_at"] = generated_at
        market = market[["trade_date", "horizon_d", "label_base", "market_ret", "benchmark_peer_count", "generated_at"]]

    industry_cols = ["trade_date", "industry_code", "horizon_d", "label_base", "industry_ret", "benchmark_peer_count"]
    if not set(industry_cols).issubset(labels.columns):
        return market, pd.DataFrame()
    industry = (
        labels[industry_cols]
        .dropna(subset=["industry_ret"])
        .drop_duplicates(subset=["trade_date", "industry_code", "horizon_d", "label_base"])
        .copy()
    )
    if not industry.empty:
        industry["generated_at"] = generated_at
        industry = industry[
            ["trade_date", "industry_code", "horizon_d", "label_base", "industry_ret", "benchmark_peer_count", "generated_at"]
        ]
    return market, industry
