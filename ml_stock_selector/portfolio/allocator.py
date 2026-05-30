from __future__ import annotations

import pandas as pd


def allocate_weights(
    selected: pd.DataFrame,
    min_weight: float,
    max_weight: float,
    allow_cash: bool,
) -> pd.DataFrame:
    out = selected.copy()
    if out.empty:
        return out
    weight = min(max(1.0 / len(out), min_weight), max_weight)
    out["target_weight"] = weight
    if not allow_cash:
        out["target_weight"] = out["target_weight"] / out["target_weight"].sum()
    return out

