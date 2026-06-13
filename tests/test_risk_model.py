from __future__ import annotations

import json

from ml_stock_selector.constants import FEATURE_SET_BASELINE_A
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.models.config import LightGBMRiskConfig
from ml_stock_selector.models.risk_model import load_risk_model, train_risk_model
from ml_stock_selector.sample_builder import build_training_samples
from ml_stock_selector.tradeability import build_tradeability_mart
from tests.ml_fixtures import create_vpa_db, normalized_bars


def test_risk_model_trains_saves_and_predicts(tmp_path):
    bars = normalized_bars()
    feature_mart = build_feature_mart(str(create_vpa_db(tmp_path / "vpa.duckdb")), bars, "2024-01-02", "2024-01-08", FEATURE_SET_BASELINE_A, [5], build_tradeability_mart(bars))
    samples = build_training_samples(feature_mart, build_labels(bars, [1]), FEATURE_SET_BASELINE_A, 1, "from_next_open")

    artifact = train_risk_model(samples, FEATURE_SET_BASELINE_A, "risk_label", "from_next_open", 1, tmp_path)
    model = load_risk_model(artifact)
    probs = model.predict_proba(samples)

    assert len(probs) == len(samples)
    assert probs.between(0.0, 1.0).all()
    assert artifact.model_type == "risk_model"
    assert artifact.model_id.startswith("risk_model_")
    assert "roc_auc" in artifact.metrics or "roc_auc_unavailable" in artifact.metrics


def test_risk_model_writes_configured_lightgbm_params_artifact(tmp_path):
    bars = normalized_bars()
    feature_mart = build_feature_mart(str(create_vpa_db(tmp_path / "vpa.duckdb")), bars, "2024-01-02", "2024-01-08", FEATURE_SET_BASELINE_A, [5], build_tradeability_mart(bars))
    samples = build_training_samples(feature_mart, build_labels(bars, [1]), FEATURE_SET_BASELINE_A, 1, "from_next_open")

    artifact = train_risk_model(
        samples,
        FEATURE_SET_BASELINE_A,
        "risk_label",
        "from_next_open",
        1,
        tmp_path,
        train_config=LightGBMRiskConfig(n_estimators=9, num_leaves=13, class_weight="balanced"),
    )

    params = json.loads(artifact.artifact_uri.with_suffix(".params.json").read_text(encoding="utf-8"))
    assert params["n_estimators"] == 9
    assert params["num_leaves"] == 13
    assert params["class_weight"] == "balanced"
