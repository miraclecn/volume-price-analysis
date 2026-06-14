from __future__ import annotations

import pandas as pd


def summarize_model_health_rows(
    nav: pd.DataFrame,
    *,
    model_or_bundle_id: str,
    strategy_id: str,
    score_version: str,
    short_window: int = 20,
    long_window: int = 60,
    max_drawdown_threshold: float = -0.20,
) -> pd.DataFrame:
    if nav.empty or "nav" not in nav:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "model_or_bundle_id",
                "strategy_id",
                "score_version",
                "rolling_20d_return",
                "rolling_60d_return",
                "rolling_20d_drawdown",
                "rolling_60d_drawdown",
                "equity_above_ma60",
                "enabled_by_health",
                "reason",
            ]
        )
    frame = nav.copy()
    date_col = "sim_date" if "sim_date" in frame else "trade_date"
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
    frame = frame.dropna(subset=[date_col, "nav"]).sort_values(date_col).reset_index(drop=True)
    records: list[dict[str, object]] = []
    for index, row in frame.iterrows():
        short = frame.iloc[max(0, index - short_window + 1) : index + 1]["nav"]
        long = frame.iloc[max(0, index - long_window + 1) : index + 1]["nav"]
        rolling_short_drawdown = _window_drawdown(short)
        rolling_long_drawdown = _window_drawdown(long)
        ma_long = float(long.mean()) if not long.empty else 0.0
        equity_above_ma = bool(float(row["nav"]) >= ma_long) if ma_long > 0.0 else True
        enabled, reason = _health_state(
            rolling_long_drawdown=rolling_long_drawdown,
            equity_above_ma=equity_above_ma,
            threshold=max_drawdown_threshold,
        )
        records.append(
            {
                "trade_date": row[date_col].strftime("%Y-%m-%d"),
                "model_or_bundle_id": model_or_bundle_id,
                "strategy_id": strategy_id,
                "score_version": score_version,
                "rolling_20d_return": _window_return(short),
                "rolling_60d_return": _window_return(long),
                "rolling_20d_drawdown": rolling_short_drawdown,
                "rolling_60d_drawdown": rolling_long_drawdown,
                "equity_above_ma60": equity_above_ma,
                "enabled_by_health": enabled,
                "reason": reason,
            }
        )
    out = pd.DataFrame(records)
    if not out.empty:
        out["equity_above_ma60"] = out["equity_above_ma60"].astype(object)
        out["enabled_by_health"] = out["enabled_by_health"].astype(object)
    return out


def _window_return(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    start = float(values.iloc[0])
    if start <= 0.0:
        return 0.0
    return round(float(values.iloc[-1] / start - 1.0), 12)


def _window_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    drawdown = values.astype(float) / values.astype(float).cummax() - 1.0
    return round(float(drawdown.min()), 12)


def _health_state(*, rolling_long_drawdown: float, equity_above_ma: bool, threshold: float) -> tuple[bool, str]:
    if rolling_long_drawdown < threshold:
        return False, "rolling_drawdown_breached"
    if not equity_above_ma:
        return False, "below_long_ma"
    return True, "healthy"
