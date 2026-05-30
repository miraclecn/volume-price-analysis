from __future__ import annotations

import pytest

from ml_stock_selector.registry import activate_model, get_active_model, register_model
from ml_stock_selector.storage import init_ml_db


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

