from __future__ import annotations

import pandas as pd


FUTURE_RETURN_HORIZONS = [1, 3, 5, 10, 20]


def compute_validation_metrics(
    states: pd.DataFrame,
    stock_prices: pd.DataFrame,
    sector_returns: pd.DataFrame | None = None,
    market_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    prices_by_code = {
        code: group.sort_values("date").reset_index(drop=True)
        for code, group in stock_prices.groupby("code", sort=False)
    }
    rows = []
    for state in states.itertuples(index=False):
        row = state._asdict()
        prices = prices_by_code.get(state.scope_id)
        metrics = _empty_metrics()
        if prices is not None:
            metrics.update(_metrics_for_state(state, prices))
            metrics.update(
                _relative_metrics(row, metrics, prices, sector_returns, market_returns)
            )
        row.update(metrics)
        rows.append(row)

    output = pd.DataFrame(rows)
    return output.astype(object).where(pd.notna(output), None)


def _metrics_for_state(state: object, prices: pd.DataFrame) -> dict[str, object]:
    matches = prices.index[prices["date"] == state.date].tolist()
    if not matches:
        return _empty_metrics()
    idx = matches[0]
    current = float(prices.loc[idx, "close"])
    metrics = _empty_metrics()
    for horizon in FUTURE_RETURN_HORIZONS:
        target_idx = idx + horizon
        if target_idx < len(prices):
            metrics[f"future_ret_{horizon}d"] = _ret(current, prices.loc[target_idx, "close"])

    for horizon in [10, 20]:
        future = prices.iloc[idx + 1 : idx + horizon + 1]
        if not future.empty:
            metrics[f"future_max_gain_{horizon}d"] = _ret(current, future["high"].max())
            metrics[f"future_max_drawdown_{horizon}d"] = _ret(current, future["low"].min())

    future_20 = prices.iloc[idx + 1 : idx + 21]
    if not future_20.empty:
        metrics["hit_new_high_20d"] = bool(future_20["high"].max() > prices.loc[idx, "high"])
        metrics["hit_new_low_20d"] = bool(future_20["low"].min() < prices.loc[idx, "low"])
    return metrics


def _relative_metrics(
    state_row: dict[str, object],
    metrics: dict[str, object],
    prices: pd.DataFrame,
    sector_returns: pd.DataFrame | None,
    market_returns: pd.DataFrame | None,
) -> dict[str, object]:
    available_10d_ret = metrics.get("future_ret_10d")
    if available_10d_ret is None:
        available_10d_ret = _available_return(prices, state_row["date"], 10)

    output = {}
    if sector_returns is not None and "sector_id" in state_row and state_row.get("sector_id"):
        sector_match = sector_returns[
            (sector_returns["date"] == state_row["date"])
            & (sector_returns["sector_id"] == state_row["sector_id"])
        ]
        if not sector_match.empty and available_10d_ret is not None:
            output["outperform_sector_10d"] = bool(
                available_10d_ret > sector_match.iloc[0]["future_ret_10d"]
            )
    if market_returns is not None:
        market_match = market_returns[market_returns["date"] == state_row["date"]]
        if not market_match.empty and available_10d_ret is not None:
            output["outperform_market_10d"] = bool(
                available_10d_ret > market_match.iloc[0]["future_ret_10d"]
            )
    return output


def _available_return(prices: pd.DataFrame, date: str, horizon: int) -> float | None:
    matches = prices.index[prices["date"] == date].tolist()
    if not matches:
        return None
    idx = matches[0]
    future = prices.iloc[idx + 1 : idx + horizon + 1]
    if future.empty:
        return None
    return _ret(prices.loc[idx, "close"], future.iloc[-1]["close"])


def _ret(current: float, future: float) -> float:
    return round(float(future) / float(current) - 1.0, 12)


def _empty_metrics() -> dict[str, object]:
    metrics: dict[str, object] = {}
    for horizon in FUTURE_RETURN_HORIZONS:
        metrics[f"future_ret_{horizon}d"] = None
    for horizon in [10, 20]:
        metrics[f"future_max_gain_{horizon}d"] = None
        metrics[f"future_max_drawdown_{horizon}d"] = None
    metrics["hit_new_high_20d"] = None
    metrics["hit_new_low_20d"] = None
    metrics["outperform_sector_10d"] = None
    metrics["outperform_market_10d"] = None
    return metrics
