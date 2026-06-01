from __future__ import annotations

import pandas as pd

from ml_stock_selector.universe import apply_universe_filter


def build_training_samples(
    feature_mart: pd.DataFrame,
    labels: pd.DataFrame,
    feature_set_id: str,
    horizon_d: int,
    label_base: str,
    label_name: str = "rank_label",
    exclude_bse: bool = False,
) -> pd.DataFrame:
    left = feature_mart[feature_mart["feature_set_id"] == feature_set_id]
    right = labels[(labels["horizon_d"] == horizon_d) & (labels["label_base"] == label_base)]
    samples = left.merge(right, on=["trade_date", "code"], how="inner")
    samples = apply_universe_filter(samples, exclude_bse=exclude_bse)
    required = _required_label_columns(label_name)
    missing = [column for column in required if column not in samples.columns]
    if missing:
        raise ValueError(f"Missing required sample columns for {label_name}: {', '.join(missing)}")
    return samples.dropna(subset=required).reset_index(drop=True)


def _required_label_columns(label_name: str) -> list[str]:
    if label_name == "rank_label":
        return ["rank_label", "future_score"]
    if label_name == "absolute_label":
        return ["absolute_label", "absolute_ret"]
    if label_name == "active_label":
        return ["active_label", "active_score"]
    if label_name == "risk_label":
        return ["risk_label"]
    return [label_name]
