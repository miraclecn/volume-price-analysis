from __future__ import annotations

from ml_stock_selector.registry import register_model
from ml_stock_selector.storage import init_ml_db


def test_register_model_records_run_fold_and_feature_store_version(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")

    register_model(
        con,
        model_id="model",
        model_type="alpha_ranker",
        feature_set_id="vpa_d_sequence",
        label_name="absolute_label",
        label_base="from_next_open",
        horizon_d=5,
        artifact_uri="model.pkl",
        feature_schema_uri="feature_schema.json",
        params_json='{"n_estimators": 7}',
        run_id="run",
        fold_id="wf_2020",
        feature_store_version="v2_pv_only_001",
    )

    row = con.execute(
        """
        select run_id, fold_id, feature_store_version, params_json, is_active
        from ml_model_registry
        where model_id = 'model'
        """
    ).fetchone()
    con.close()

    assert row == ("run", "wf_2020", "v2_pv_only_001", '{"n_estimators": 7}', False)
