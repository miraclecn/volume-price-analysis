from __future__ import annotations

import math

import pandas as pd

from ml_stock_selector.portfolio.constraints import is_unknown_industry


def max_drawdown(nav: pd.DataFrame, nav_col: str = "nav", date_col: str = "sim_date") -> float:
    values = _ordered_nav_values(nav, nav_col, date_col)
    if values.empty:
        return 0.0
    drawdown = values / values.cummax() - 1.0
    return float(drawdown.min())


def annualized_return(
    nav: pd.DataFrame,
    nav_col: str = "nav",
    periods_per_year: int = 252,
    date_col: str = "sim_date",
) -> float:
    values = _ordered_nav_values(nav, nav_col, date_col)
    if len(values) < 2:
        return 0.0
    start = float(values.iloc[0])
    end = float(values.iloc[-1])
    if start <= 0.0:
        return 0.0
    if end <= 0.0:
        return -1.0
    years = _observed_years(nav, date_col, len(values), periods_per_year)
    if years <= 0.0:
        return 0.0
    return float((end / start) ** (1.0 / years) - 1.0)


def _ordered_nav_values(nav: pd.DataFrame, nav_col: str, date_col: str) -> pd.Series:
    if nav.empty or nav_col not in nav:
        return pd.Series(dtype=float)
    frame = nav.copy()
    if date_col in frame:
        frame = frame.sort_values(date_col)
    return pd.to_numeric(frame[nav_col], errors="coerce").dropna().astype(float)


def _observed_years(nav: pd.DataFrame, date_col: str, observation_count: int, periods_per_year: int) -> float:
    if date_col in nav:
        dates = pd.to_datetime(nav.sort_values(date_col)[date_col], errors="coerce").dropna()
        if len(dates) >= 2:
            days = (dates.iloc[-1] - dates.iloc[0]).days
            if days > 0:
                return days / 365.25
    return (observation_count - 1) / periods_per_year


def cash_days_ratio(nav: pd.DataFrame, exposure_col: str = "gross_exposure") -> float:
    if nav.empty or exposure_col not in nav:
        return 0.0
    exposure = pd.to_numeric(nav[exposure_col], errors="coerce").fillna(0.0)
    return float((exposure <= 0.0).mean())


def pool_size_metrics(candidate_pool: pd.DataFrame, core_pool: pd.DataFrame) -> dict[str, float]:
    return {
        "candidate_pool_size": float(len(candidate_pool)),
        "core_pool_size": float(len(core_pool)),
    }


def holding_period_metrics(orders: pd.DataFrame, nav: pd.DataFrame | None = None) -> dict[str, float]:
    if orders.empty or "side" not in orders:
        holding_days = pd.Series(dtype=float)
    else:
        sells = orders[(orders["side"].astype(str).str.lower() == "sell") & (orders.get("status", "filled") == "filled")]
        holding_days = pd.to_numeric(sells.get("holding_days", pd.Series(dtype=float)), errors="coerce").dropna()
    turnover_daily_avg = 0.0
    if nav is not None and not nav.empty and "turnover" in nav:
        turnover_daily_avg = float(pd.to_numeric(nav["turnover"], errors="coerce").fillna(0.0).mean())
    sell_blocked_count = 0.0
    if not orders.empty and "side" in orders:
        sell_orders = orders[orders["side"].astype(str).str.lower() == "sell"]
        sell_blocked_count = float((sell_orders.get("status", "filled") != "filled").sum())
    return {
        "avg_holding_days": float(holding_days.mean()) if not holding_days.empty else 0.0,
        "median_holding_days": float(holding_days.median()) if not holding_days.empty else 0.0,
        "max_holding_days": float(holding_days.max()) if not holding_days.empty else 0.0,
        "holding_segment_count": float(len(holding_days)),
        "turnover_daily_avg": turnover_daily_avg,
        "sell_blocked_count": sell_blocked_count,
    }


def summarize_fold_metric_rows(
    result,
    *,
    run_id: str,
    fold_id: str,
    score_version: str,
    strategy_id: str,
    start_date: str,
    end_date: str,
    candidate_pool_size: float,
    core_pool_size: float,
    bse_excluded_count: float = 0.0,
) -> pd.DataFrame:
    nav = result.nav
    ordered_nav = _ordered_nav_values(nav, "nav", "sim_date")
    total_return = 0.0
    if len(ordered_nav) >= 2 and float(ordered_nav.iloc[0]) > 0.0:
        total_return = float(ordered_nav.iloc[-1] / ordered_nav.iloc[0] - 1.0)
    annual_return = annualized_return(nav)
    drawdown = max_drawdown(nav)
    calmar = float(annual_return / abs(drawdown)) if drawdown < 0.0 else 0.0
    turnover = float(pd.to_numeric(nav.get("turnover", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()) if not nav.empty else 0.0
    returns = ordered_nav.pct_change().dropna()
    win_rate = float((returns > 0.0).mean()) if len(returns) else 0.0
    empty_day_ratio = cash_days_ratio(nav)
    if nav.empty or "gross_exposure" not in nav:
        cash_ratio_avg = 0.0
    else:
        exposure = pd.to_numeric(nav["gross_exposure"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        cash_ratio_avg = float((1.0 - exposure).mean())
    if result.positions.empty or "industry_code" not in result.positions:
        unknown = {"unknown_industry_avg_weight": 0.0}
    else:
        unknown = summarize_unknown_industry_exposure(unknown_industry_daily_exposure(result.positions, result.orders))
    holding = holding_period_metrics(result.orders, nav)
    values = {
        "annual_return": annual_return,
        "total_return": total_return,
        "max_drawdown": drawdown,
        "calmar_like": calmar,
        "turnover": turnover,
        "win_rate": win_rate,
        "empty_day_ratio": empty_day_ratio,
        "cash_ratio_avg": cash_ratio_avg,
        "bse_excluded_count": float(bse_excluded_count),
        "unknown_industry_weight_avg": float(unknown["unknown_industry_avg_weight"]),
        "core_pool_size_avg": float(core_pool_size),
        "candidate_pool_size_avg": float(candidate_pool_size),
        **holding,
    }
    context = {
        "run_id": run_id,
        "fold_id": fold_id,
        "strategy_id": strategy_id,
        "score_version": score_version,
        "start_date": start_date,
        "end_date": end_date,
        "segment": "fold",
        **values,
    }
    return pd.DataFrame(
        [
            {
                **context,
                "metric_name": name,
                "metric_value": value,
            }
            for name, value in values.items()
        ]
    )


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
