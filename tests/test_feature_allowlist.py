from __future__ import annotations

import json

import pandas as pd

from ml_stock_selector.feature_store import export_feature_store
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


def test_feature_allowlist_prevents_new_fields_from_entering_training_schema(tmp_path):
    allowlist = tmp_path / "allowlist.yaml"
    allowlist.write_text(
        """
feature_set_id: vpa_d_sequence
schema_version: v2_pv_only_001
include_patterns:
  - "ret_*"
  - "volume_ratio_*"
exclude_columns:
  - industry_code
  - is_bse
""".strip(),
        encoding="utf-8",
    )
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
                        "volume_ratio_5d": 1.5,
                        "range_pct": 0.02,
                        "unexpected_new_feature": 9.0,
                        "industry_code": "I1",
                        "is_bse": False,
                    }
                ),
            }
        ]
    )
    upsert_dataframe(con, "ml_feature_mart_daily", rows, ["trade_date", "code", "feature_set_id"])

    result = export_feature_store(
        con,
        output_dir=tmp_path / "feature_store",
        dataset_version="v2_pv_only_001",
        feature_set_id="vpa_d_sequence",
        start_date="2024-01-01",
        end_date="2024-01-31",
        allowlist_path=allowlist,
    )

    schema = json.loads(result.feature_schema_path.read_text(encoding="utf-8"))
    assert schema["numeric_columns"] == ["ret_1d", "volume_ratio_5d"]
