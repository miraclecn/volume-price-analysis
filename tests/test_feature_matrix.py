from __future__ import annotations

import json

import pandas as pd

from ml_stock_selector.feature_matrix import build_feature_matrix, load_feature_schema, save_feature_schema


def test_feature_matrix_schema_roundtrip_aligns_inference_columns(tmp_path):
    train = pd.DataFrame(
        {
            "features_json": [
                json.dumps({"x": 1.0, "cat": "A"}),
                json.dumps({"x": 2.0, "cat": "B"}),
            ]
        }
    )
    matrix, schema = build_feature_matrix(train, "set", fit=True)
    path = tmp_path / "schema.json"
    save_feature_schema(schema, path)
    loaded = load_feature_schema(path)

    inference = pd.DataFrame({"features_json": [json.dumps({"x": None, "cat": "Z"})]})
    inferred, _ = build_feature_matrix(inference, "set", schema=loaded, fit=False)

    assert list(inferred.columns) == schema.output_columns
    assert inferred.iloc[0]["x"] == 0.0
    assert "cat=__UNKNOWN__" in inferred.columns
    assert matrix.shape[1] == inferred.shape[1]

