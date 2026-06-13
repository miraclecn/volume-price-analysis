from __future__ import annotations

import json

import pandas as pd
import pytest

from ml_stock_selector.feature_store import export_feature_store
from ml_stock_selector.feature_store_reader import FeatureStoreSpec, load_feature_schema
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


def test_feature_store_metadata_records_schema_hash_and_reader_rejects_mismatch(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    rows = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "feature_set_id": "vpa_d_sequence",
                "generated_at": "t",
                "features_json": json.dumps({"ret_1d": 0.1, "range_pct": 0.02}),
            },
            {
                "trade_date": "2024-01-05",
                "code": "000002.SZ",
                "feature_set_id": "vpa_d_sequence",
                "generated_at": "t",
                "features_json": json.dumps({"ret_1d": 0.2, "range_pct": 0.03}),
            },
        ]
    )
    upsert_dataframe(con, "ml_feature_mart_daily", rows, ["trade_date", "code", "feature_set_id"])

    result = export_feature_store(
        con,
        output_dir=tmp_path / "feature_store",
        dataset_version="v2",
        feature_set_id="vpa_d_sequence",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    schema_payload = json.loads(result.feature_schema_path.read_text(encoding="utf-8"))
    metadata_payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert schema_payload["schema_hash"]
    assert metadata_payload["schema_hash"] == schema_payload["schema_hash"]
    assert metadata_payload["min_date"] == "2024-01-02"
    assert metadata_payload["max_date"] == "2024-01-05"
    assert metadata_payload["source_db"].endswith("ml.duckdb")

    schema = load_feature_schema(FeatureStoreSpec(str(tmp_path / "feature_store"), "v2", "vpa_d_sequence"))
    assert schema.schema_hash == schema_payload["schema_hash"]

    schema_payload["numeric_columns"].append("unexpected_new_feature")
    result.feature_schema_path.write_text(json.dumps(schema_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_hash"):
        load_feature_schema(FeatureStoreSpec(str(tmp_path / "feature_store"), "v2", "vpa_d_sequence"))
