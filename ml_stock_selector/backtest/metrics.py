from __future__ import annotations

import math

import pandas as pd

from ml_stock_selector.portfolio.constraints import is_unknown_industry


def max_drawdown(nav: pd.DataFrame, nav_col: str = "nav") -> float:
    values = nav[nav_col].astype(float)
    drawdown = values / values.cummax() - 1.0
    return float(drawdown.min())


def annualized_return(nav: pd.DataFrame, nav_col: str = "nav", periods_per_year: int = 252) -> float:
    values = nav[nav_col].astype(float)
    if len(values) < 2:
        return 0.0
    returns = values.pct_change().dropna()
    if returns.empty:
        return 0.0
    return float((1.0 + returns.mean()) ** periods_per_year - 1.0)


def rank_ic(frame: pd.DataFrame, pred_col: str, label_col: str) -> float:
    return float(frame[pred_col].rank().corr(frame[label_col].rank()))


def ndcg_at_k(frame: pd.DataFrame, pred_col: str, label_col: str, k: int) -> float:
    ordered = frame.sort_values(pred_col, ascending=False).head(k)
    ideal = frame.sort_values(label_col, ascending=False).head(k)
    dcg = _dcg(ordered[label_col].tolist())
    idcg = _dcg(ideal[label_col].tolist())
    return float(dcg / idcg) if idcg else 0.0


def unknown_industry_daily_exposure(
    positions: pd.DataFrame,
    orders: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame(
            columns=[
                "sim_date",
                "unknown_industry_position_count",
                "unknown_industry_weight",
                "unknown_industry_trade_count",
            ]
        )
    frame = positions.copy()
    frame["_industry_unknown"] = frame["industry_code"].map(is_unknown_industry)
    daily_positions = (
        frame[frame["_industry_unknown"]]
        .groupby("sim_date", sort=True)
        .agg(
            unknown_industry_position_count=("code", "nunique"),
            unknown_industry_weight=("weight", "sum"),
        )
        .reset_index()
    )
    all_dates = pd.DataFrame({"sim_date": sorted(frame["sim_date"].dropna().unique())})
    daily = all_dates.merge(daily_positions, on="sim_date", how="left").fillna(
        {
            "unknown_industry_position_count": 0,
            "unknown_industry_weight": 0.0,
        }
    )
    daily["unknown_industry_position_count"] = daily[
        "unknown_industry_position_count"
    ].astype(int)
    daily["unknown_industry_trade_count"] = 0
    if orders is not None and not orders.empty and "industry_code" in orders:
        unknown_orders = orders[
            orders["industry_code"].map(is_unknown_industry)
            & (orders.get("status", "filled") == "filled")
        ]
        trades = (
            unknown_orders.groupby("sim_date", sort=True)
            .size()
            .rename("unknown_industry_trade_count")
            .reset_index()
        )
        daily = daily.drop(columns=["unknown_industry_trade_count"]).merge(
            trades, on="sim_date", how="left"
        )
        daily["unknown_industry_trade_count"] = daily[
            "unknown_industry_trade_count"
        ].fillna(0).astype(int)
    return daily


def summarize_unknown_industry_exposure(daily_exposure: pd.DataFrame) -> dict[str, float]:
    if daily_exposure.empty:
        return {
            "unknown_industry_max_weight": 0.0,
            "unknown_industry_avg_weight": 0.0,
            "unknown_industry_days": 0.0,
            "unknown_industry_trade_count": 0.0,
        }
    weights = pd.to_numeric(
        daily_exposure["unknown_industry_weight"], errors="coerce"
    ).fillna(0.0)
    return {
        "unknown_industry_max_weight": float(weights.max()),
        "unknown_industry_avg_weight": float(weights.mean()),
        "unknown_industry_days": float((weights > 0).sum()),
        "unknown_industry_trade_count": float(
            daily_exposure.get("unknown_industry_trade_count", pd.Series(dtype=float)).sum()
        ),
    }


def _dcg(values: list[float]) -> float:
    return sum((2**float(value) - 1.0) / math.log2(idx + 2) for idx, value in enumerate(values))
