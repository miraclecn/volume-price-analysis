from __future__ import annotations

import json

import pandas as pd

from ml_stock_selector.feature_store import write_feature_frame_to_feature_store


def test_feature_mart_frame_can_write_partitioned_feature_store_directly(tmp_path):
    feature_mart = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "feature_set_id": "vpa_d_sequence",
                "features_json": json.dumps({"ret_1d": 0.1, "range_pct": 0.02}),
            }
        ]
    )

    result = write_feature_frame_to_feature_store(
        feature_mart,
        output_dir=tmp_path / "feature_store",
        dataset_version="v2",
        feature_set_id="vpa_d_sequence",
        source_db="direct_feature_mart",
    )

    assert result.row_count == 1
    assert result.metadata_path.exists()
    assert list((result.root_dir / "year=2024" / "month=01").glob("*.parquet"))
