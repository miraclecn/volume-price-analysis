from __future__ import annotations

import pytest

from ml_stock_selector.constants import MODEL_TYPE_ACTIVE_RANKER, MODEL_TYPE_RANKER, MODEL_TYPE_RISK
from ml_stock_selector.registry import (
    activate_model_bundle,
    get_active_model_bundle,
    get_active_model,
    register_model,
    register_model_bundle,
)
from ml_stock_selector.storage import init_ml_db


def _register_three_models(con, suffix: str) -> tuple[str, str, str]:
    absolute = f"abs_{suffix}"
    active = f"act_{suffix}"
    risk = f"risk_{suffix}"
    for model_id, model_type, label_name in [
        (absolute, MODEL_TYPE_RANKER, "absolute_label"),
        (active, MODEL_TYPE_ACTIVE_RANKER, "active_label"),
        (risk, MODEL_TYPE_RISK, "risk_label"),
    ]:
        register_model(
            con,
            model_id=model_id,
            model_type=model_type,
            feature_set_id="vpa_d_sequence",
            label_name=label_name,
            label_base="from_next_open",
            horizon_d=5,
            artifact_uri=f"{model_id}.pkl",
            feature_schema_uri="schema.json",
        )
    return absolute, active, risk


def test_model_bundle_activation_retires_previous_bundle_and_activates_models(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    first = _register_three_models(con, "first")
    second = _register_three_models(con, "second")
    register_model_bundle(
        con,
        bundle_id="bundle_first",
        run_id="run_first",
        bundle_role="production",
        absolute_model_id=first[0],
        active_model_id=first[1],
        risk_model_id=first[2],
        feature_set_id="vpa_d_sequence",
        label_base="from_next_open",
        horizon_d=5,
        score_version="v2_three_model",
        artifact_dir=str(tmp_path / "bundle_first"),
    )
    register_model_bundle(
        con,
        bundle_id="bundle_second",
        run_id="run_second",
        bundle_role="production",
        absolute_model_id=second[0],
        active_model_id=second[1],
        risk_model_id=second[2],
        feature_set_id="vpa_d_sequence",
        label_base="from_next_open",
        horizon_d=5,
        score_version="v2_three_model",
        artifact_dir=str(tmp_path / "bundle_second"),
    )

    activate_model_bundle(con, "bundle_first")
    activate_model_bundle(con, "bundle_second")

    bundle = get_active_model_bundle(con, "production", "vpa_d_sequence", "from_next_open", 5)
    statuses = con.execute(
        "select bundle_id, status from ml_model_bundles order by bundle_id"
    ).fetchall()
    active_abs = get_active_model(con, MODEL_TYPE_RANKER, "vpa_d_sequence", "absolute_label", "from_next_open", 5)
    con.close()

    assert bundle["bundle_id"] == "bundle_second"
    assert statuses == [("bundle_first", "retired"), ("bundle_second", "active")]
    assert active_abs["model_id"] == "abs_second"


def test_activate_model_bundle_rejects_missing_component_model(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    first = _register_three_models(con, "first")
    register_model_bundle(
        con,
        bundle_id="bad_bundle",
        run_id="run_bad",
        bundle_role="production",
        absolute_model_id=first[0],
        active_model_id="missing_active",
        risk_model_id=first[2],
        feature_set_id="vpa_d_sequence",
        label_base="from_next_open",
        horizon_d=5,
        score_version="v2_three_model",
        artifact_dir=str(tmp_path / "bad_bundle"),
    )

    with pytest.raises(ValueError, match="missing_active"):
        activate_model_bundle(con, "bad_bundle")
    con.close()
