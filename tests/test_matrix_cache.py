from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ml_stock_selector.feature_store import export_feature_store
from ml_stock_selector.feature_store_reader import FeatureStoreSpec
from ml_stock_selector.matrix_cache import build_fold_matrix_cache, load_train_matrix
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


def test_build_fold_matrix_cache_joins_labels_metadata_and_excludes_bse(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    feature_rows = []
    label_rows = []
    tradeability_rows = []
    for date in ["2024-01-02", "2024-01-03", "2024-01-04"]:
        for code in ["000001.SZ", "000002.SZ", "920001.BJ"]:
            feature_rows.append(
                {
                    "trade_date": date,
                    "code": code,
                    "feature_set_id": "vpa_d_sequence",
                    "generated_at": "t",
                    "features_json": json.dumps({"ret_1d": 0.1, "range_pct": 0.02}),
                }
            )
            label_rows.append(
                {
                    "trade_date": date,
                    "code": code,
                    "horizon_d": 5,
                    "label_base": "from_next_open",
                    "absolute_label": 1,
                    "active_label": 2,
                    "risk_label": 0,
                    "generated_at": "t",
                }
            )
            tradeability_rows.append(
                {
                    "trade_date": date,
                    "code": code,
                    "industry_code": "I1",
                    "industry_name": "Industry",
                    "is_bse": code.endswith(".BJ"),
                    "is_st": False,
                    "is_paused": False,
                    "adv20_amount": 100.0,
                    "can_buy_next_open": code != "000002.SZ",
                    "generated_at": "t",
                }
            )
    upsert_dataframe(con, "ml_feature_mart_daily", pd.DataFrame(feature_rows), ["trade_date", "code", "feature_set_id"])
    upsert_dataframe(con, "ml_labels_daily", pd.DataFrame(label_rows), ["trade_date", "code", "horizon_d", "label_base"])
    upsert_dataframe(con, "ml_tradeability_daily", pd.DataFrame(tradeability_rows), ["trade_date", "code"])
    export_feature_store(con, tmp_path / "feature_store", "v2", "vpa_d_sequence", "2024-01-01", "2024-01-31")

    cache = build_fold_matrix_cache(
        con,
        FeatureStoreSpec(str(tmp_path / "feature_store"), "v2", "vpa_d_sequence"),
        SimpleNamespace(
            fold_id="wf_test",
            train_start="2024-01-02",
            train_end="2024-01-02",
            valid_start="2024-01-03",
            valid_end="2024-01-03",
            test_start="2024-01-04",
            test_end="2024-01-04",
        ),
        run_id="run",
        feature_set_id="vpa_d_sequence",
        horizon_d=5,
        label_base="from_next_open",
        universe_config={"exclude_bse": True},
        cache_root=tmp_path / "cache",
        batch_size=2,
    )

    x_train = load_train_matrix(cache)
    y_abs = np.load(cache.y_abs_train_path)
    group = np.load(cache.group_train_path)
    metadata = con.execute("select * from read_parquet(?)", [str(cache.metadata_train_path)]).fetchdf()
    manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))

    assert x_train.shape == (1, 2)
    assert x_train.dtype == np.float32
    assert y_abs.shape == (1,)
    assert group.sum() == x_train.shape[0]
    assert metadata["is_bse"].fillna(False).eq(False).all()
    assert metadata["can_buy_next_open"].fillna(False).eq(True).all()
    assert set(metadata["code"]) == {"000001.SZ"}
    assert manifest["train_rows"] == 1
