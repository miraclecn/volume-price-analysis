from __future__ import annotations

import pandas as pd

from ml_stock_selector.models.artifacts import ModelArtifact
from ml_stock_selector.prediction import build_three_model_prediction_rows


def _artifact(model_id: str, model_type: str) -> ModelArtifact:
    from pathlib import Path

    return ModelArtifact(model_id, model_type, "vpa_d_sequence", "x", "from_next_open", 1, Path("a"), Path("b"), Path("."), {})


def test_three_model_prediction_rows_have_model_lineage():
    feature_mart = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-02"],
            "code": ["000001.SZ", "000002.SZ"],
            "feature_set_id": ["vpa_d_sequence", "vpa_d_sequence"],
        }
    )
    rows = build_three_model_prediction_rows(
        feature_mart,
        pd.Series([0.1, 0.2]),
        pd.Series([0.3, 0.4]),
        pd.Series([0.2, 0.1]),
        _artifact("abs_1", "alpha_ranker"),
        _artifact("act_1", "active_ranker"),
        _artifact("risk_1", "risk_model"),
    )
    assert rows["model_id"].str.contains("three_model").all()
    assert rows["score_version"].eq("v2_three_model").all()

