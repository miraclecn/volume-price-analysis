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
    buy_count = 0.0
    sell_count = 0.0
    risk_exit_count = 0.0
    time_exit_count = 0.0
    avg_entry_abs_rank_pct = 0.0
    avg_entry_risk_rank_pct = 0.0
    if not orders.empty and "side" in orders:
        filled = orders[orders.get("status", "filled") == "filled"]
        buy_count = float(((filled["side"].astype(str).str.lower() == "buy")).sum())
        sell_orders = orders[orders["side"].astype(str).str.lower() == "sell"]
        filled_sells = sell_orders[sell_orders.get("status", "filled") == "filled"]
        sell_count = float(len(filled_sells))
        sell_blocked_count = float((sell_orders.get("status", "filled") != "filled").sum())
        if "exit_reason" in filled_sells:
            risk_exit_count = float((filled_sells["exit_reason"] == "risk_exit").sum())
            time_exit_count = float((filled_sells["exit_reason"] == "time_exit").sum())
        avg_entry_abs_rank_pct = _avg_column(filled, "entry_abs_rank_pct")
        avg_entry_risk_rank_pct = _avg_column(filled, "entry_risk_rank_pct")
    return {
        "avg_holding_days": float(holding_days.mean()) if not holding_days.empty else 0.0,
        "median_holding_days": float(holding_days.median()) if not holding_days.empty else 0.0,
        "max_holding_days": float(holding_days.max()) if not holding_days.empty else 0.0,
        "holding_segment_count": float(len(holding_days)),
        "turnover_daily_avg": turnover_daily_avg,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "risk_exit_count": risk_exit_count,
        "time_exit_count": time_exit_count,
        "sell_blocked_count": sell_blocked_count,
        "avg_entry_abs_rank_pct": avg_entry_abs_rank_pct,
        "avg_entry_risk_rank_pct": avg_entry_risk_rank_pct,
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
    returns = _daily_returns(nav)
    monthly_returns = _monthly_returns(nav)
    win_rate = _win_rate(returns)
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
        "calmar": calmar,
        "calmar_like": calmar,
        "sharpe": _sharpe(returns),
        "sortino": _sortino(returns),
        "volatility": _annualized_volatility(returns),
        "turnover": turnover,
        "win_rate": win_rate,
        "win_rate_daily": win_rate,
        "win_rate_monthly": _win_rate(monthly_returns),
        "empty_day_ratio": empty_day_ratio,
        "cash_ratio_avg": cash_ratio_avg,
        "avg_cash_ratio": cash_ratio_avg,
        "position_count_avg": _avg_positions(result.positions),
        "avg_positions": _avg_positions(result.positions),
        "best_month": _series_max(monthly_returns),
        "worst_month": _series_min(monthly_returns),
        "max_consecutive_loss_days": _max_consecutive_losses(returns),
        "max_consecutive_loss_months": _max_consecutive_losses(monthly_returns),
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


def summarize_walkforward_metric_rows(
    metrics: pd.DataFrame,
    *,
    run_id: str,
    score_version: str | None = None,
    strategy_id: str | None = None,
    high_return_threshold: float = 1.0,
) -> pd.DataFrame:
    fold_metrics = _fold_metric_pivot(metrics, run_id=run_id, score_version=score_version, strategy_id=strategy_id)
    returns = pd.to_numeric(fold_metrics.get("annual_return", pd.Series(dtype=float)), errors="coerce").dropna()
    drawdowns = pd.to_numeric(fold_metrics.get("max_drawdown", pd.Series(dtype=float)), errors="coerce").dropna()
    calmar = pd.to_numeric(
        fold_metrics.get("calmar", fold_metrics.get("calmar_like", pd.Series(dtype=float))),
        errors="coerce",
    ).dropna()
    worst_fold = _extreme_fold(fold_metrics, "annual_return", "min")
    best_fold = _extreme_fold(fold_metrics, "annual_return", "max")
    high_capture = _high_return_capture_ratio(
        metrics,
        fold_metrics,
        run_id=run_id,
        score_version=score_version,
        strategy_id=strategy_id,
        high_return_threshold=high_return_threshold,
    )
    values = {
        "mean_annual_return": _series_mean(returns),
        "median_annual_return": _series_median(returns),
        "min_annual_return": _series_min(returns),
        "max_annual_return": _series_max(returns),
        "std_annual_return": _series_std(returns),
        "positive_year_ratio": _win_rate(returns),
        "max_of_max_drawdown": _series_min(drawdowns),
        "mean_drawdown": _series_mean(drawdowns),
        "mean_calmar": _series_mean(calmar),
        "worst_year": _fold_year_value(worst_fold),
        "best_year": _fold_year_value(best_fold),
        "negative_year_count": float((returns < 0.0).sum()) if not returns.empty else 0.0,
        "worst_year_return": _series_min(returns),
        "best_year_return": _series_max(returns),
        "drawdown_over_20_count": float((drawdowns <= -0.20).sum()) if not drawdowns.empty else 0.0,
        "drawdown_over_30_count": float((drawdowns <= -0.30).sum()) if not drawdowns.empty else 0.0,
        "high_return_capture_ratio": high_capture,
        "aggressive_year_capture_ratio": high_capture,
    }
    return _metric_rows(
        values,
        run_id=run_id,
        fold_id="all",
        score_version=score_version or "all",
        strategy_id=strategy_id,
        start_date=None,
        end_date=None,
        segment="walkforward",
    )


def compare_backtest_metric_rows(
    metrics: pd.DataFrame,
    *,
    run_id: str,
    fold_id: str | None = None,
) -> pd.DataFrame:
    values: dict[str, float] = {}
    _add_delta(
        values,
        "absolute_only_vs_three_model_delta",
            _metric_value(metrics, run_id, "annual_return", score_version="v2_absolute_only", fold_id=fold_id),
            _metric_value(metrics, run_id, "annual_return", score_version="v2_three_model", strategy_id="holding_aware_v2", fold_id=fold_id),
    )
    _add_delta(
        values,
        "score_mode_return_delta",
            _metric_value(metrics, run_id, "annual_return", score_version="v2_three_model", strategy_id="holding_aware_v2", fold_id=fold_id),
            _metric_value(metrics, run_id, "annual_return", score_version="v2_absolute_only", fold_id=fold_id),
    )
    _add_delta(
        values,
        "score_mode_drawdown_delta",
            _metric_value(metrics, run_id, "max_drawdown", score_version="v2_three_model", strategy_id="holding_aware_v2", fold_id=fold_id),
            _metric_value(metrics, run_id, "max_drawdown", score_version="v2_absolute_only", fold_id=fold_id),
    )
    _add_delta(
        values,
        "risk_filter_return_delta",
            _metric_value(metrics, run_id, "annual_return", score_version="v2_absolute_risk_filter", fold_id=fold_id),
            _metric_value(metrics, run_id, "annual_return", score_version="v2_absolute_only", fold_id=fold_id),
    )
    _add_delta(
        values,
        "risk_filter_drawdown_delta",
            _metric_value(metrics, run_id, "max_drawdown", score_version="v2_absolute_risk_filter", fold_id=fold_id),
            _metric_value(metrics, run_id, "max_drawdown", score_version="v2_absolute_only", fold_id=fold_id),
    )
    _add_delta(
        values,
        "absolute_risk_sort_vs_risk_filter_delta",
            _metric_value(metrics, run_id, "annual_return", score_version="v2_absolute_risk_sort", fold_id=fold_id),
            _metric_value(metrics, run_id, "annual_return", score_version="v2_absolute_risk_filter", fold_id=fold_id),
    )
    _add_delta(
        values,
        "risk_exit_benefit",
            _metric_value(metrics, run_id, "annual_return", strategy_id="abs_ranker_fixed_5d_risk_filter_v1", fold_id=fold_id),
            _metric_value(metrics, run_id, "annual_return", strategy_id="abs_ranker_fixed_5d_no_risk_exit_v1", fold_id=fold_id),
    )
    _add_delta(
        values,
        "fixed_horizon_vs_holding_aware_delta",
            _metric_value(metrics, run_id, "annual_return", strategy_id="abs_ranker_fixed_5d_risk_filter_v1", fold_id=fold_id),
            _metric_value(metrics, run_id, "annual_return", strategy_id="holding_aware_v2", score_version="v2_three_model", fold_id=fold_id),
    )
    _add_delta(
        values,
        "fixed_horizon_vs_holding_aware_drawdown_delta",
            _metric_value(metrics, run_id, "max_drawdown", strategy_id="abs_ranker_fixed_5d_risk_filter_v1", fold_id=fold_id),
            _metric_value(metrics, run_id, "max_drawdown", strategy_id="holding_aware_v2", score_version="v2_three_model", fold_id=fold_id),
    )
    return _metric_rows(
        values,
        run_id=run_id,
        fold_id=fold_id or "all",
        score_version="comparison",
        strategy_id=None,
        start_date=None,
        end_date=None,
        segment="comparison",
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


def _daily_returns(nav: pd.DataFrame, nav_col: str = "nav", date_col: str = "sim_date") -> pd.Series:
    values = _ordered_nav_values(nav, nav_col, date_col)
    if len(values) < 2:
        return pd.Series(dtype=float)
    return values.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna().astype(float)


def _monthly_returns(nav: pd.DataFrame, nav_col: str = "nav", date_col: str = "sim_date") -> pd.Series:
    if nav.empty or nav_col not in nav or date_col not in nav:
        return pd.Series(dtype=float)
    frame = nav[[date_col, nav_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[nav_col] = pd.to_numeric(frame[nav_col], errors="coerce")
    frame = frame.dropna().sort_values(date_col)
    if len(frame) < 2:
        return pd.Series(dtype=float)
    frame["_period"] = frame[date_col].dt.to_period("M")
    monthly = (
        frame.groupby("_period", sort=True)[nav_col]
        .agg(first="first", last="last")
        .reset_index()
    )
    returns: list[float] = []
    previous_month_close: float | None = None
    for row in monthly.itertuples(index=False):
        first_value = float(row.first)
        last_value = float(row.last)
        if previous_month_close is not None and previous_month_close > 0.0:
            returns.append(first_value / previous_month_close - 1.0)
        if first_value > 0.0 and first_value != last_value:
            returns.append(last_value / first_value - 1.0)
        previous_month_close = last_value
    return pd.Series(returns, dtype=float)


def _annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if len(values) < 2:
        return 0.0
    return _clean_float(float(values.std(ddof=0) * math.sqrt(periods_per_year)))


def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if len(values) < 2:
        return 0.0
    volatility = float(values.std(ddof=0))
    if volatility <= 0.0:
        return 0.0
    return _clean_float(float(values.mean() / volatility * math.sqrt(periods_per_year)))


def _sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    downside = values[values < 0.0]
    if len(values) < 2 or len(downside) < 1:
        return 0.0
    downside_deviation = float((downside.pow(2).mean()) ** 0.5)
    if downside_deviation <= 0.0:
        return 0.0
    return _clean_float(float(values.mean() / downside_deviation * math.sqrt(periods_per_year)))


def _win_rate(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return _clean_float(float((clean > 0.0).mean())) if not clean.empty else 0.0


def _max_consecutive_losses(values: pd.Series) -> float:
    longest = 0
    current = 0
    for value in pd.to_numeric(values, errors="coerce").dropna():
        if float(value) < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest)


def _series_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return _clean_float(float(clean.mean())) if not clean.empty else 0.0


def _series_median(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return _clean_float(float(clean.median())) if not clean.empty else 0.0


def _series_min(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return _clean_float(float(clean.min())) if not clean.empty else 0.0


def _series_max(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return _clean_float(float(clean.max())) if not clean.empty else 0.0


def _series_std(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return _clean_float(float(clean.std(ddof=0))) if len(clean) >= 2 else 0.0


def _metric_rows(
    values: dict[str, float],
    *,
    run_id: str,
    fold_id: str | None,
    score_version: str,
    strategy_id: str | None,
    start_date: str | None,
    end_date: str | None,
    segment: str,
) -> pd.DataFrame:
    clean_values = {name: _clean_float(value) for name, value in values.items()}
    context = {
        "run_id": run_id,
        "fold_id": fold_id,
        "strategy_id": strategy_id,
        "score_version": score_version,
        "start_date": start_date,
        "end_date": end_date,
        "segment": segment,
        **clean_values,
    }
    return pd.DataFrame(
        [
            {
                **context,
                "metric_name": name,
                "metric_value": value,
            }
            for name, value in clean_values.items()
        ]
    )


def _fold_metric_pivot(
    metrics: pd.DataFrame,
    *,
    run_id: str,
    score_version: str | None,
    strategy_id: str | None,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    frame = metrics[metrics["run_id"].astype(str) == str(run_id)].copy()
    if "segment" in frame:
        frame = frame[frame["segment"].astype(str) == "fold"]
    if score_version is not None and "score_version" in frame:
        frame = frame[frame["score_version"].astype(str) == str(score_version)]
    if strategy_id is not None and "strategy_id" in frame:
        frame = frame[frame["strategy_id"].astype(str) == str(strategy_id)]
    if frame.empty:
        return pd.DataFrame()
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    pivot = frame.pivot_table(index="fold_id", columns="metric_name", values="metric_value", aggfunc="mean")
    return pivot.reset_index()


def _extreme_fold(fold_metrics: pd.DataFrame, metric_name: str, direction: str) -> str | None:
    if fold_metrics.empty or metric_name not in fold_metrics or "fold_id" not in fold_metrics:
        return None
    values = pd.to_numeric(fold_metrics[metric_name], errors="coerce")
    if values.dropna().empty:
        return None
    index = values.idxmin() if direction == "min" else values.idxmax()
    return str(fold_metrics.loc[index, "fold_id"])


def _fold_year_value(fold_id: str | None) -> float:
    if not fold_id:
        return 0.0
    digits = "".join(char for char in str(fold_id) if char.isdigit())
    if len(digits) >= 4:
        return float(digits[-4:])
    return 0.0


def _high_return_capture_ratio(
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    run_id: str,
    score_version: str | None,
    strategy_id: str | None,
    high_return_threshold: float,
) -> float:
    if fold_metrics.empty or "annual_return" not in fold_metrics:
        return 0.0
    selected = fold_metrics[["fold_id", "annual_return"]].copy()
    selected["annual_return"] = pd.to_numeric(selected["annual_return"], errors="coerce")
    baseline = _aggressive_baseline_returns(metrics, run_id=run_id, score_version=score_version, strategy_id=strategy_id)
    if baseline.empty:
        high = selected[selected["annual_return"] >= high_return_threshold]
        return 1.0 if not high.empty else 0.0
    joined = selected.merge(baseline, on="fold_id", how="inner", suffixes=("", "_aggressive"))
    joined = joined[pd.to_numeric(joined["annual_return_aggressive"], errors="coerce") >= high_return_threshold]
    if joined.empty:
        return 0.0
    captures = joined["annual_return"] / joined["annual_return_aggressive"]
    return _series_mean(captures)


def _aggressive_baseline_returns(
    metrics: pd.DataFrame,
    *,
    run_id: str,
    score_version: str | None,
    strategy_id: str | None,
) -> pd.DataFrame:
    if metrics.empty or "metric_name" not in metrics:
        return pd.DataFrame(columns=["fold_id", "annual_return"])
    frame = metrics[
        (metrics["run_id"].astype(str) == str(run_id))
        & (metrics["metric_name"].astype(str) == "annual_return")
    ].copy()
    if "segment" in frame:
        frame = frame[frame["segment"].astype(str) == "fold"]
    identity = (
        frame.get("score_version", pd.Series("", index=frame.index)).astype(str)
        + " "
        + frame.get("strategy_id", pd.Series("", index=frame.index)).astype(str)
    )
    frame = frame[identity.str.contains("aggressive", case=False, na=False)]
    if score_version is not None and "score_version" in frame:
        frame = frame[frame["score_version"].astype(str) != str(score_version)]
    if strategy_id is not None and "strategy_id" in frame:
        frame = frame[frame["strategy_id"].astype(str) != str(strategy_id)]
    if frame.empty:
        return pd.DataFrame(columns=["fold_id", "annual_return"])
    frame["annual_return"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    return frame.groupby("fold_id", as_index=False)["annual_return"].mean()


def _metric_value(
    metrics: pd.DataFrame,
    run_id: str,
    metric_name: str,
    *,
    score_version: str | None = None,
    strategy_id: str | None = None,
    fold_id: str | None = None,
) -> float | None:
    if metrics.empty:
        return None
    frame = metrics[
        (metrics["run_id"].astype(str) == str(run_id))
        & (metrics["metric_name"].astype(str) == str(metric_name))
    ].copy()
    if "segment" in frame:
        frame = frame[frame["segment"].astype(str) == "fold"]
    if score_version is not None and "score_version" in frame:
        frame = frame[frame["score_version"].astype(str) == str(score_version)]
    if strategy_id is not None and "strategy_id" in frame:
        frame = frame[frame["strategy_id"].astype(str) == str(strategy_id)]
    if fold_id is not None and "fold_id" in frame:
        frame = frame[frame["fold_id"].astype(str) == str(fold_id)]
    values = pd.to_numeric(frame.get("metric_value", pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return None
    return _clean_float(float(values.mean()))


def _add_delta(values: dict[str, float], name: str, left: float | None, right: float | None) -> None:
    if left is None or right is None:
        return
    values[name] = _clean_float(left - right)


def _clean_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return round(result, 12)


def _avg_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def _avg_positions(positions: pd.DataFrame) -> float:
    if positions.empty or "sim_date" not in positions or "code" not in positions:
        return 0.0
    return float(positions.groupby("sim_date")["code"].nunique().mean())
