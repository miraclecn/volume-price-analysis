from __future__ import annotations

import pandas as pd


def backtest_strategy_allocation(
    allocation: pd.DataFrame,
    sleeve_nav: pd.DataFrame,
    *,
    strategy_id: str = "phase9_ensemble_v1",
    score_version: str = "strategy_allocation_v1",
) -> pd.DataFrame:
    if allocation.empty:
        return pd.DataFrame(columns=["sim_date", "strategy_id", "score_version", "nav", "cash", "gross_exposure", "turnover"])
    alloc = allocation.copy()
    alloc["trade_date"] = pd.to_datetime(alloc["trade_date"], errors="coerce")
    alloc["final_weight"] = pd.to_numeric(alloc["final_weight"], errors="coerce").fillna(0.0)
    returns = _sleeve_returns(sleeve_nav)
    merged = alloc.merge(
        returns,
        on=["trade_date", "strategy_id", "score_version"],
        how="left",
    )
    merged["daily_return"] = merged["daily_return"].fillna(0.0)
    merged["weighted_return"] = merged["final_weight"] * merged["daily_return"]
    merged["non_cash_weight"] = merged["final_weight"].where(merged["sleeve"] != "cash", 0.0)
    daily = (
        merged.groupby("trade_date", sort=True)
        .agg(
            daily_return=("weighted_return", "sum"),
            gross_exposure=("non_cash_weight", "sum"),
        )
        .reset_index()
    )
    daily["nav"] = (1.0 + daily["daily_return"]).cumprod().round(12)
    daily["cash"] = (1.0 - daily["gross_exposure"]).clip(lower=0.0).round(12)
    daily["turnover"] = 0.0
    daily["strategy_id"] = strategy_id
    daily["score_version"] = score_version
    daily["sim_date"] = daily["trade_date"].dt.strftime("%Y-%m-%d")
    return daily[["sim_date", "strategy_id", "score_version", "nav", "cash", "gross_exposure", "turnover"]]


def _sleeve_returns(sleeve_nav: pd.DataFrame) -> pd.DataFrame:
    if sleeve_nav.empty:
        return pd.DataFrame(columns=["trade_date", "strategy_id", "score_version", "daily_return"])
    frame = sleeve_nav.copy()
    date_col = "sim_date" if "sim_date" in frame else "trade_date"
    frame["trade_date"] = pd.to_datetime(frame[date_col], errors="coerce")
    frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "nav"])
    key_cols = ["strategy_id", "score_version"]
    frame = frame.sort_values(key_cols + ["trade_date"])
    frame["daily_return"] = frame.groupby(key_cols)["nav"].pct_change().fillna(0.0)
    return frame[["trade_date", "strategy_id", "score_version", "daily_return"]]
