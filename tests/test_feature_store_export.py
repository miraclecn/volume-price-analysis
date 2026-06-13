from __future__ import annotations

import json

import pandas as pd

from ml_stock_selector.feature_store import export_feature_store
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


def test_export_feature_store_writes_partitioned_numeric_parquet_without_metadata(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    rows = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "feature_set_id": "vpa_d_sequence",
                "generated_at": "t",
                "features_json": json.dumps(
                    {
                        "ret_1d": 0.1,
                        "is_breakout": True,
                        "sequence_pattern": "UP",
                        "industry_code": "I1",
                        "is_bse": False,
                    }
                ),
            },
            {
                "trade_date": "2024-02-02",
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
        dataset_version="v2_pv_only_001",
        feature_set_id="vpa_d_sequence",
        start_date="2024-01-01",
        end_date="2024-12-31",
        chunk_size=1,
    )

    schema = json.loads(result.feature_schema_path.read_text(encoding="utf-8"))
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    parquet_dir = (
        tmp_path
        / "feature_store"
        / "dataset_version=v2_pv_only_001"
        / "feature_set_id=vpa_d_sequence"
        / "year=2024"
        / "month=01"
    )

    assert list(parquet_dir.glob("*.parquet"))
    assert metadata["row_count"] == 2
    assert schema["numeric_columns"] == ["is_breakout", "range_pct", "ret_1d"]

    exported = con.execute(
        "select * from read_parquet(?) order by code",
        [str(tmp_path / "feature_store" / "dataset_version=v2_pv_only_001" / "feature_set_id=vpa_d_sequence" / "year=*" / "month=*" / "*.parquet")],
    ).fetchdf()
    assert list(exported["code"]) == ["000001.SZ", "000002.SZ"]
    assert "industry_code" not in exported.columns
    assert "is_bse" not in exported.columns
    assert "sequence_pattern" not in exported.columns
    assert exported["ret_1d"].dtype == "float32"
