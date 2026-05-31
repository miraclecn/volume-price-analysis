from __future__ import annotations

from ml_stock_selector.constants import FEATURE_SET_BASELINE_A, MODEL_TYPE_ACTIVE_RANKER
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.models.active_ranker import load_active_ranker, train_active_ranker
from ml_stock_selector.sample_builder import build_training_samples
from ml_stock_selector.tradeability import build_tradeability_mart
from tests.ml_fixtures import create_vpa_db, normalized_bars


def test_active_ranker_trains_as_independent_model_role(tmp_path):
    bars = normalized_bars()
    feature_mart = build_feature_mart(str(create_vpa_db(tmp_path / "vpa.duckdb")), bars, "2024-01-02", "2024-01-08", FEATURE_SET_BASELINE_A, [5], build_tradeability_mart(bars))
    samples = build_training_samples(
        feature_mart,
        build_labels(bars, [1], include_v2=True),
        FEATURE_SET_BASELINE_A,
        1,
        "from_next_open",
        label_name="active_label",
    )

    artifact = train_active_ranker(samples, FEATURE_SET_BASELINE_A, "active_label", "from_next_open", 1, tmp_path)
    model = load_active_ranker(artifact)

    assert artifact.model_type == MODEL_TYPE_ACTIVE_RANKER
    assert artifact.label_name == "active_label"
    assert len(model.predict(samples)) == len(samples)
