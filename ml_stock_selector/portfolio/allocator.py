from __future__ import annotations

import pandas as pd


def allocate_weights(
    selected: pd.DataFrame,
    min_weight: float,
    max_weight: float,
    allow_cash: bool,
) -> pd.DataFrame:
    attrs = dict(selected.attrs)
    out = selected.copy()
    out.attrs.update(attrs)
    if out.empty:
        return out
    if "trade_date" in out and out["trade_date"].nunique(dropna=False) > 1:
        weighted_days = [
            allocate_weights(day, min_weight, max_weight, allow_cash)
            for _, day in out.groupby("trade_date", sort=True)
        ]
        concat_frames = []
        for frame in weighted_days:
            clean = frame.copy()
            clean.attrs.clear()
            concat_frames.append(clean)
        weighted = pd.concat(concat_frames, ignore_index=True)
        weighted.attrs.update(attrs)
        return weighted
    active_mask = pd.Series(True, index=out.index)
    if "signal_action" in out:
        active_mask &= out["signal_action"].astype(str) != "sell"
    active_count = int(active_mask.sum())
    if active_count == 0:
        out["target_weight"] = 0.0
        out.attrs.update(attrs)
        return out
    weight = min(max(1.0 / active_count, min_weight), max_weight)
    out["target_weight"] = 0.0
    out.loc[active_mask, "target_weight"] = weight
    if not allow_cash:
        total = out["target_weight"].sum()
        if total:
            out["target_weight"] = out["target_weight"] / total
    out.attrs.update(attrs)
    return out
