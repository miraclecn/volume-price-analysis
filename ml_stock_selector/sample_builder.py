from __future__ import annotations

import pandas as pd


def build_training_samples(
    feature_mart: pd.DataFrame,
    labels: pd.DataFrame,
    feature_set_id: str,
    horizon_d: int,
    label_base: str,
) -> pd.DataFrame:
    left = feature_mart[feature_mart["feature_set_id"] == feature_set_id]
    right = labels[(labels["horizon_d"] == horizon_d) & (labels["label_base"] == label_base)]
    samples = left.merge(right, on=["trade_date", "code"], how="inner")
    return samples.dropna(subset=["rank_label", "future_score"]).reset_index(drop=True)

