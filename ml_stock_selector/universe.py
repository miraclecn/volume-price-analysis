from __future__ import annotations

import pandas as pd


def detect_is_bse(code: object) -> bool:
    if code is None or pd.isna(code):
        return False
    value = str(code).strip().upper()
    return value.endswith(".BJ") or value.startswith("BJ")


def with_is_bse(frame: pd.DataFrame, code_column: str = "code") -> pd.DataFrame:
    out = frame.copy()
    out["is_bse"] = out[code_column].map(detect_is_bse)
    return out


def apply_universe_filter(
    frame: pd.DataFrame,
    *,
    exclude_bse: bool = False,
    code_column: str = "code",
) -> pd.DataFrame:
    out = with_is_bse(frame, code_column=code_column)
    if exclude_bse:
        out = out[~out["is_bse"].fillna(False).astype(bool)].copy()
    return out
