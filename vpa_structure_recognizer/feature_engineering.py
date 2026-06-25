from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "date",
    "scope_type",
    "scope_id",
    "window_n",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "amount",
    "ret_pct",
    "range_pct",
    "body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "close_position",
    "vol_ma_n",
    "vol_rvol_n",
    "range_pct_ma_n",
    "range_rvol_n",
    "body_pct_ma_n",
    "body_rvol_n",
    "price_high_n",
    "price_low_n",
    "prev_price_high_n",
    "prev_price_low_n",
    "price_position_n",
    "ma_n",
    "ma_slope_n",
]


def compute_features(
    bars: pd.DataFrame,
    windows: list[int],
    scope_type: str,
    scope_id_column: str | None = None,
    scope_id: str | None = None,
) -> pd.DataFrame:
    base = bars.copy()
    if scope_id_column is not None:
        base["scope_id"] = base[scope_id_column]
    elif scope_id is not None:
        base["scope_id"] = scope_id
    else:
        raise ValueError("scope_id_column or scope_id is required")

    base["scope_type"] = scope_type
    for column in ["open", "high", "low", "close", "prev_close", "volume", "amount"]:
        base[column] = pd.to_numeric(base[column], errors="coerce")
    base = base.sort_values(["scope_id", "date"]).reset_index(drop=True)
    _add_base_price_features(base)

    window_frames = []
    for window in windows:
        features = base.copy()
        grouped = features.groupby("scope_id", sort=False)
        features["window_n"] = int(window)
        features["vol_ma_n"] = _rolling(grouped["volume"], window, "mean")
        features["vol_rvol_n"] = _safe_div(features["volume"], features["vol_ma_n"])
        features["range_pct_ma_n"] = _rolling(grouped["range_pct"], window, "mean")
        features["range_rvol_n"] = _safe_div(
            features["range_pct"], features["range_pct_ma_n"]
        )
        features["body_pct_ma_n"] = _rolling(grouped["body_pct"], window, "mean")
        features["body_rvol_n"] = _safe_div(features["body_pct"], features["body_pct_ma_n"])
        features["price_high_n"] = _rolling(grouped["high"], window, "max")
        features["price_low_n"] = _rolling(grouped["low"], window, "min")
        features["prev_price_high_n"] = _rolling_prior(grouped["high"], window, "max")
        features["prev_price_low_n"] = _rolling_prior(grouped["low"], window, "min")
        features["price_position_n"] = _safe_div(
            features["close"] - features["price_low_n"],
            features["price_high_n"] - features["price_low_n"],
        )
        features["ma_n"] = _rolling(grouped["close"], window, "mean")
        previous_ma = grouped["ma_n"].shift(1)
        features["ma_slope_n"] = _safe_div(features["ma_n"] - previous_ma, previous_ma)
        window_frames.append(features[FEATURE_COLUMNS])

    output = pd.concat(window_frames, ignore_index=True)
    output = output.sort_values(["scope_id", "date", "window_n"]).reset_index(drop=True)
    return output


def _add_base_price_features(frame: pd.DataFrame) -> None:
    price_range = frame["high"] - frame["low"]
    body_abs = (frame["close"] - frame["open"]).abs()
    upper_shadow = frame["high"] - frame[["open", "close"]].max(axis=1)
    lower_shadow = frame[["open", "close"]].min(axis=1) - frame["low"]

    frame["ret_pct"] = _safe_div(frame["close"], frame["prev_close"]) - 1
    frame["range_pct"] = _safe_div(price_range, frame["prev_close"])
    frame["body_pct"] = _safe_div(body_abs, frame["prev_close"])
    frame["upper_shadow_pct"] = _safe_div(upper_shadow, frame["prev_close"])
    frame["lower_shadow_pct"] = _safe_div(lower_shadow, frame["prev_close"])
    frame["body_ratio"] = _safe_div(body_abs, price_range).fillna(0.0)
    frame["upper_shadow_ratio"] = _safe_div(upper_shadow, price_range).fillna(0.0)
    frame["lower_shadow_ratio"] = _safe_div(lower_shadow, price_range).fillna(0.0)
    frame["close_position"] = _safe_div(frame["close"] - frame["low"], price_range)


def _rolling(grouped: pd.core.groupby.SeriesGroupBy, window: int, op: str) -> pd.Series:
    if op == "mean":
        return grouped.transform(lambda values: values.rolling(window, min_periods=1).mean())
    if op == "max":
        return grouped.transform(lambda values: values.rolling(window, min_periods=1).max())
    if op == "min":
        return grouped.transform(lambda values: values.rolling(window, min_periods=1).min())
    raise ValueError(f"unsupported rolling operation: {op}")


def _rolling_prior(grouped: pd.core.groupby.SeriesGroupBy, window: int, op: str) -> pd.Series:
    if op == "max":
        return grouped.transform(lambda values: values.rolling(window, min_periods=1).max().shift(1))
    if op == "min":
        return grouped.transform(lambda values: values.rolling(window, min_periods=1).min().shift(1))
    raise ValueError(f"unsupported rolling operation: {op}")


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numeric_numerator = pd.to_numeric(numerator, errors="coerce")
    numeric_denominator = pd.to_numeric(denominator, errors="coerce")
    valid = numeric_denominator.notna() & (numeric_denominator != 0)
    result = pd.Series(index=numeric_numerator.index, dtype="float64")
    result.loc[valid] = numeric_numerator.loc[valid] / numeric_denominator.loc[valid]
    return result
