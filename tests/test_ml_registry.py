from __future__ import annotations

import pytest

from ml_stock_selector.constants import FEATURE_SCHEMA_V2_NO_INDUSTRY, FEATURE_SET_BASELINE_A, MODEL_TYPE_ACTIVE_RANKER, MODEL_TYPE_RANKER, MODEL_TYPE_RISK
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.registry import activate_model, get_active_model, register_model
from ml_stock_selector.feature_matrix import load_feature_schema
from ml_stock_selector.storage import init_ml_db
from ml_stock_selector.tradeability import build_tradeability_mart
from scripts.train_ml_models import train_model_artifacts
from tests.ml_fixtures import create_vpa_db, normalized_bars


def test_model_registry_activation_deactivates_same_key(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    register_model(con, model_id="m1", model_type="alpha_ranker", feature_set_id="set", label_name="rank_label", label_base="from_next_open", horizon_d=5, artifact_uri="a", feature_schema_uri="s")
    register_model(con, model_id="m2", model_type="alpha_ranker", feature_set_id="set", label_name="rank_label", label_base="from_next_open", horizon_d=5, artifact_uri="b", feature_schema_uri="s")
    activate_model(con, "m1")
    activate_model(con, "m2")

    active = get_active_model(con, "alpha_ranker", "set", "rank_label", "from_next_open", 5)

    assert active["model_id"] == "m2"
    with pytest.raises(ValueError, match="active"):
        get_active_model(con, "risk_model", "set", "rank_label", "from_next_open", 5)
    con.close()


def test_train_model_artifacts_builds_three_v2_roles(tmp_path):
    bars = normalized_bars()
    feature_mart = build_feature_mart(str(create_vpa_db(tmp_path / "vpa.duckdb")), bars, "2024-01-02", "2024-01-08", FEATURE_SET_BASELINE_A, [5], build_tradeability_mart(bars))
    labels = build_labels(bars, [1], include_v2=True)

    artifacts = train_model_artifacts(
        feature_mart,
        labels,
        FEATURE_SET_BASELINE_A,
        1,
        "from_next_open",
        tmp_path,
        {
            "labels_v2_enabled": True,
            "feature_matrix_v2_deny_industry": True,
            "active_ranker_enabled": True,
            "risk_model_v2_enabled": True,
        },
    )

    assert {artifact.model_type for artifact in artifacts} == {
        MODEL_TYPE_RANKER,
        MODEL_TYPE_ACTIVE_RANKER,
        MODEL_TYPE_RISK,
    }
    assert {artifact.label_name for artifact in artifacts} == {"absolute_label", "active_label", "risk_label"}
    schemas = [load_feature_schema(artifact.feature_schema_uri) for artifact in artifacts]
    assert {schema.schema_version for schema in schemas} == {FEATURE_SCHEMA_V2_NO_INDUSTRY}
    for schema in schemas:
        assert all(not column.startswith("industry") for column in schema.output_columns)
