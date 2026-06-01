from __future__ import annotations

import pandas as pd

from ml_stock_selector.sample_builder import build_training_samples


def test_training_samples_exclude_bse():
    feature_mart = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-02"],
            "code": ["430001.BJ", "000001.SZ"],
            "feature_set_id": ["vpa_d_sequence", "vpa_d_sequence"],
            "x": [1.0, 2.0],
        }
    )
    labels = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-02"],
            "code": ["430001.BJ", "000001.SZ"],
            "horizon_d": [1, 1],
            "label_base": ["from_next_open", "from_next_open"],
            "rank_label": [1, 1],
            "future_score": [0.1, 0.2],
        }
    )
    samples = build_training_samples(
        feature_mart,
        labels,
        "vpa_d_sequence",
        1,
        "from_next_open",
        exclude_bse=True,
    )
    assert samples["code"].tolist() == ["000001.SZ"]

