from __future__ import annotations

import math

import pandas as pd


LIMIT_BAND_UNKNOWN = "unknown"
LIMIT_BAND_OTHER = "other"


def add_limit_band_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if not {"limit_up", "limit_down", "prev_close"}.issubset(out.columns):
        out["limit_up_pct"] = None
        out["limit_down_pct"] = None
        out["limit_band"] = LIMIT_BAND_UNKNOWN
        return out

    prev_close = pd.to_numeric(out["prev_close"], errors="coerce")
    valid_prev_close = prev_close.where(prev_close > 0)
    out["limit_up_pct"] = pd.to_numeric(out["limit_up"], errors="coerce") / valid_prev_close - 1.0
    out["limit_down_pct"] = pd.to_numeric(out["limit_down"], errors="coerce") / valid_prev_close - 1.0
    band_width = pd.concat(
        [out["limit_up_pct"].abs(), out["limit_down_pct"].abs()],
        axis=1,
    ).mean(axis=1, skipna=True)
    out["limit_band"] = band_width.map(classify_limit_band)
    return out


def classify_limit_band(width: object, tolerance: float = 0.015) -> str:
    try:
        value = float(width)
    except (TypeError, ValueError):
        return LIMIT_BAND_UNKNOWN
    if not math.isfinite(value):
        return LIMIT_BAND_UNKNOWN
    known = {
        0.05: "limit_5pct",
        0.10: "limit_10pct",
        0.20: "limit_20pct",
        0.30: "limit_30pct",
    }
    nearest = min(known, key=lambda item: abs(value - item))
    if abs(value - nearest) <= tolerance:
        return known[nearest]
    return LIMIT_BAND_OTHER
