from __future__ import annotations

import json

import pandas as pd

from ml_stock_selector.feature_store import export_feature_store
from ml_stock_selector.feature_store_reader import FeatureStoreSpec, iter_feature_store_batches, load_feature_schema
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


def test_feature_store_reader_filters_dates_and_preserves_schema_order(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    rows = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "feature_set_id": "vpa_d_sequence",
                "generated_at": "t",
                "features_json": json.dumps({"z_feature": 1.0, "a_feature": 2.0}),
            },
            {
                "trade_date": "2024-02-02",
                "code": "000002.SZ",
                "feature_set_id": "vpa_d_sequence",
                "generated_at": "t",
                "features_json": json.dumps({"z_feature": 3.0, "a_feature": 4.0}),
            },
        ]
    )
    upsert_dataframe(con, "ml_feature_mart_daily", rows, ["trade_date", "code", "feature_set_id"])
    export_feature_store(
        con,
        output_dir=tmp_path / "feature_store",
        dataset_version="v2",
        feature_set_id="vpa_d_sequence",
        start_date="2024-01-01",
        end_date="2024-12-31",
        chunk_size=10,
    )

    spec = FeatureStoreSpec(str(tmp_path / "feature_store"), "v2", "vpa_d_sequence")
    schema = load_feature_schema(spec)
    batches = list(iter_feature_store_batches(spec, "2024-02-01", "2024-02-28", batch_size=1))

    assert schema.numeric_columns == ["a_feature", "z_feature"]
    assert len(batches) == 1
    assert list(batches[0].columns) == ["trade_date", "code", "a_feature", "z_feature"]
    assert batches[0]["code"].tolist() == ["000002.SZ"]

