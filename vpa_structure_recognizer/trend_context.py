from __future__ import annotations

import pandas as pd


TREND_CONTEXT_COLUMNS = [
    "date",
    "scope_type",
    "scope_id",
    "window_n",
    "parent_window_n",
    "parent_high",
    "parent_low",
    "parent_price_position",
    "parent_ma",
    "parent_ma_slope",
    "trend_label",
    "position_label",
    "trend_strength_score",
]


def compute_trend_context(
    features: pd.DataFrame,
    parent_windows: dict[int, list[int]],
) -> pd.DataFrame:
    indexed = {
        (row.date, row.scope_type, row.scope_id, int(row.window_n)): row
        for row in features.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []

    for row in features.sort_values(["scope_type", "scope_id", "date", "window_n"]).itertuples(
        index=False
    ):
        window = int(row.window_n)
        parent_window = parent_windows.get(window, [window])[0]
        parent = indexed.get((row.date, row.scope_type, row.scope_id, parent_window))
        if parent is None:
            rows.append(_unknown_context(row, parent_window))
            continue

        parent_high = parent.price_high_n
        parent_low = parent.price_low_n
        position = _safe_position(row.close, parent_low, parent_high)
        rows.append(
            {
                "date": row.date,
                "scope_type": row.scope_type,
                "scope_id": row.scope_id,
                "window_n": window,
                "parent_window_n": parent_window,
                "parent_high": parent_high,
                "parent_low": parent_low,
                "parent_price_position": position,
                "parent_ma": parent.ma_n,
                "parent_ma_slope": parent.ma_slope_n,
                "trend_label": _trend_label(row.close, parent.ma_n, parent.ma_slope_n),
                "position_label": _position_label(position),
                "trend_strength_score": _trend_strength(position, parent.ma_slope_n),
            }
        )

    output = pd.DataFrame(rows, columns=TREND_CONTEXT_COLUMNS)
    return output.astype(object).where(pd.notna(output), None)


def _unknown_context(row: object, parent_window: int) -> dict[str, object]:
    return {
        "date": row.date,
        "scope_type": row.scope_type,
        "scope_id": row.scope_id,
        "window_n": int(row.window_n),
        "parent_window_n": parent_window,
        "parent_high": None,
        "parent_low": None,
        "parent_price_position": None,
        "parent_ma": None,
        "parent_ma_slope": None,
        "trend_label": "UNKNOWN",
        "position_label": "UNKNOWN",
        "trend_strength_score": 0.0,
    }


def _safe_position(close: float, low: float, high: float) -> float | None:
    if low is None or high is None or high == low:
        return None
    return (close - low) / (high - low)


def _position_label(position: float | None) -> str:
    if position is None:
        return "UNKNOWN"
    if position < 0.25:
        return "LOW"
    if position < 0.45:
        return "MID_LOW"
    if position < 0.65:
        return "MID"
    if position < 0.85:
        return "MID_HIGH"
    return "HIGH"


def _trend_label(close: float, parent_ma: float | None, parent_ma_slope: float | None) -> str:
    if parent_ma is None or parent_ma_slope is None:
        return "UNKNOWN"
    if parent_ma_slope > 0.01 and close >= parent_ma:
        return "UPTREND"
    if parent_ma_slope < -0.01 and close <= parent_ma:
        return "DOWNTREND"
    if parent_ma_slope > 0.005:
        return "RECOVERING"
    if parent_ma_slope < -0.005:
        return "WEAKENING"
    return "SIDEWAYS"


def _trend_strength(position: float | None, parent_ma_slope: float | None) -> float:
    if position is None or parent_ma_slope is None:
        return 0.0
    score = 50.0 + parent_ma_slope * 1000.0 + (position - 0.5) * 50.0
    return max(0.0, min(100.0, round(score, 4)))
