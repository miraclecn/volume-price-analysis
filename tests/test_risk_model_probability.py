from __future__ import annotations

import pandas as pd

from ml_stock_selector.models.artifacts import ModelArtifact
from ml_stock_selector.models.risk_model import LogisticFallbackModel
from ml_stock_selector.prediction import predict_with_model


def test_risk_model_probability_in_0_1(tmp_path):
    from pathlib import Path
    import pickle

    model = LogisticFallbackModel(["x"], {"x": 1.0}, 0.0)
    artifact_path = tmp_path / "risk.pkl"
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        "{\"feature_set_id\":\"vpa_d_sequence\",\"numeric_columns\":[\"x\"],\"categorical_columns\":[],\"output_columns\":[\"x\"],\"category_levels\":{},\"fill_values\":{\"x\":0.0},\"schema_version\":\"v1\"}",
        encoding="utf-8",
    )
    with artifact_path.open("wb") as handle:
        pickle.dump(model, handle)
    artifact = ModelArtifact("risk_1", "risk_model", "vpa_d_sequence", "risk_label", "from_next_open", 1, Path(schema_path), Path(artifact_path), Path(tmp_path), {})
    feature_mart = pd.DataFrame({"trade_date": ["2024-01-02"], "code": ["000001.SZ"], "feature_set_id": ["vpa_d_sequence"], "features_json": ['{"x": 0.5}']})
    prob = float(predict_with_model(feature_mart, artifact).iloc[0])
    assert 0.0 <= prob <= 1.0
