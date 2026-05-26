from __future__ import annotations

import pandas as pd


LABEL_COLUMNS = [
    "date",
    "scope_type",
    "scope_id",
    "window_n",
    "parent_window_n",
    "raw_label",
    "normal_or_abnormal",
    "volume_level",
    "price_result_level",
    "efficiency_level",
    "bull_bear_score",
    "supply_score",
    "demand_score",
    "volatility_score",
    "description",
]


def classify_volume_level(vol_rvol_n: float | None) -> str:
    if _missing(vol_rvol_n):
        return "UNKNOWN_VOLUME"
    if vol_rvol_n < 0.7:
        return "LOW_VOLUME"
    if vol_rvol_n < 1.2:
        return "NORMAL_VOLUME"
    if vol_rvol_n < 1.8:
        return "MILD_HIGH_VOLUME"
    if vol_rvol_n <= 2.5:
        return "HIGH_VOLUME"
    return "EXTREME_HIGH_VOLUME"


def label_bars(features: pd.DataFrame, parent_windows: dict[int, list[int]]) -> pd.DataFrame:
    rows = []
    for row in features.sort_values(["scope_type", "scope_id", "date", "window_n"]).itertuples(
        index=False
    ):
        label, normality, description = _label_row(row)
        rows.append(
            {
                "date": row.date,
                "scope_type": row.scope_type,
                "scope_id": row.scope_id,
                "window_n": int(row.window_n),
                "parent_window_n": parent_windows.get(int(row.window_n), [int(row.window_n)])[0],
                "raw_label": label,
                "normal_or_abnormal": normality,
                "volume_level": classify_volume_level(row.vol_rvol_n),
                "price_result_level": _price_result_level(row.ret_pct),
                "efficiency_level": _efficiency_level(row),
                "bull_bear_score": _bull_bear_score(row),
                "supply_score": _supply_score(label, row),
                "demand_score": _demand_score(label, row),
                "volatility_score": _volatility_score(row),
                "description": description,
            }
        )
    return pd.DataFrame(rows, columns=LABEL_COLUMNS)


def _label_row(row: object) -> tuple[str, str, str]:
    vol_rvol_n = _value(row.vol_rvol_n)
    range_rvol_n = _value(row.range_rvol_n)
    body_ratio = _value(row.body_ratio)
    upper_shadow_ratio = _value(row.upper_shadow_ratio)
    lower_shadow_ratio = _value(row.lower_shadow_ratio)
    close_position = _value(row.close_position, 0.5)
    ret_pct = _value(row.ret_pct)
    high = _value(row.high)
    low = _value(row.low)
    close = _value(row.close)
    has_price_high = not _missing(row.price_high_n)
    has_price_low = not _missing(row.price_low_n)
    price_high_n = _value(row.price_high_n)
    price_low_n = _value(row.price_low_n)
    if (
        vol_rvol_n >= 1.8
        and range_rvol_n <= 0.9
        and body_ratio <= 0.35
    ):
        return (
            "HIGH_VOLUME_LOW_PROGRESS",
            "ABNORMAL",
            "High volume produced limited price progress.",
        )
    if (
        vol_rvol_n >= 1.5
        and upper_shadow_ratio >= 0.45
        and close_position <= 0.6
    ):
        return (
            "HIGH_VOLUME_UPPER_SUPPLY",
            "ABNORMAL",
            "High volume met upper supply and closed away from the high.",
        )
    if (
        vol_rvol_n >= 1.5
        and lower_shadow_ratio >= 0.45
        and close_position >= 0.4
    ):
        return (
            "HIGH_VOLUME_LOWER_SUPPORT",
            "ABNORMAL",
            "High volume found lower support and recovered from the low.",
        )
    if (
        vol_rvol_n < 0.8
        and ret_pct > 0
        and range_rvol_n >= 1.2
        and close_position >= 0.7
    ):
        return ("LOW_VOLUME_BIG_UP", "ABNORMAL", "Large up move lacked volume support.")
    if (
        vol_rvol_n < 0.8
        and ret_pct < 0
        and range_rvol_n >= 1.2
        and close_position <= 0.3
    ):
        return (
            "LOW_VOLUME_BIG_DOWN",
            "ABNORMAL",
            "Large down move occurred on low relative volume.",
        )
    if (
        has_price_high
        and high >= price_high_n
        and close < price_high_n
        and upper_shadow_ratio >= 0.45
    ):
        return (
            "BREAKOUT_PULLBACK",
            "ABNORMAL",
            "Intraday breakout could not hold by the close.",
        )
    if (
        has_price_low
        and low <= price_low_n
        and close > price_low_n
        and lower_shadow_ratio >= 0.45
    ):
        return (
            "BREAKDOWN_RECOVERY",
            "ABNORMAL",
            "Intraday breakdown recovered by the close.",
        )
    if (
        ret_pct > 0
        and vol_rvol_n >= 1.0
        and body_ratio >= 0.45
        and close_position >= 0.65
    ):
        return (
            "NORMAL_UP_CONFIRM",
            "NORMAL",
            "Up move had volume support and a strong close.",
        )
    if (
        ret_pct < 0
        and vol_rvol_n >= 1.0
        and body_ratio >= 0.45
        and close_position <= 0.35
    ):
        return (
            "NORMAL_DOWN_CONFIRM",
            "NORMAL",
            "Down move had volume support and a weak close.",
        )
    if vol_rvol_n < 0.8 and range_rvol_n < 0.9 and abs(ret_pct) <= 0.01:
        return (
            "LOW_VOLUME_SMALL_MOVE",
            "NORMAL",
            "Low volume narrow-range session.",
        )
    return ("NEUTRAL", "NORMAL", "No dominant single-day volume-price condition.")


def _price_result_level(ret_pct: float) -> str:
    ret_pct = _value(ret_pct)
    if ret_pct > 0:
        return "UP"
    if ret_pct < 0:
        return "DOWN"
    return "FLAT"


def _efficiency_level(row: object) -> str:
    vol_rvol_n = _value(row.vol_rvol_n)
    range_rvol_n = _value(row.range_rvol_n)
    if vol_rvol_n >= 1.8 and range_rvol_n <= 0.9:
        return "LOW_EFFICIENCY"
    if vol_rvol_n < 0.8 and range_rvol_n >= 1.2:
        return "UNCONFIRMED_PROGRESS"
    return "NORMAL_EFFICIENCY"


def _bull_bear_score(row: object) -> float:
    ret_pct = _value(row.ret_pct)
    close_position = _value(row.close_position, 0.5)
    direction = 1.0 if ret_pct > 0 else -1.0 if ret_pct < 0 else 0.0
    close_bias = (close_position - 0.5) * 2.0
    return round((direction * 50.0) + (close_bias * 25.0), 4)


def _supply_score(label: str, row: object) -> float:
    score = 0.0
    if "SUPPLY" in label or "PULLBACK" in label:
        score += 70.0
    if _value(row.upper_shadow_ratio) >= 0.45:
        score += 20.0
    if _value(row.close_position, 0.5) <= 0.35:
        score += 10.0
    return min(100.0, score)


def _demand_score(label: str, row: object) -> float:
    score = 0.0
    if "SUPPORT" in label or label == "NORMAL_UP_CONFIRM":
        score += 70.0
    if _value(row.lower_shadow_ratio) >= 0.45:
        score += 20.0
    if _value(row.close_position, 0.5) >= 0.65:
        score += 10.0
    return min(100.0, score)


def _volatility_score(row: object) -> float:
    return round(min(100.0, max(0.0, _value(row.range_rvol_n) * 50.0)), 4)


def _missing(value: object) -> bool:
    return value is None or pd.isna(value)


def _value(value: object, default: float = 0.0) -> float:
    if _missing(value):
        return default
    return float(value)
