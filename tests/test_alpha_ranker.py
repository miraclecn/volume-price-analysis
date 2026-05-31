from __future__ import annotations

from ml_stock_selector.constants import FEATURE_SET_BASELINE_A
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.models.alpha_ranker import load_alpha_ranker, train_alpha_ranker
from ml_stock_selector.sample_builder import build_training_samples
from ml_stock_selector.tradeability import build_tradeability_mart
from tests.ml_fixtures import create_vpa_db, normalized_bars


def test_alpha_ranker_trains_saves_and_reloads(tmp_path):
    bars = normalized_bars()
    feature_mart = build_feature_mart(str(create_vpa_db(tmp_path / "vpa.duckdb")), bars, "2024-01-02", "2024-01-08", FEATURE_SET_BASELINE_A, [5], build_tradeability_mart(bars))
    samples = build_training_samples(feature_mart, build_labels(bars, [1]), FEATURE_SET_BASELINE_A, 1, "from_next_open")

    artifact = train_alpha_ranker(samples, FEATURE_SET_BASELINE_A, "rank_label", "from_next_open", 1, tmp_path)
    model = load_alpha_ranker(artifact)

    preds = model.predict(samples)
    assert len(preds) == len(samples)
    assert artifact.artifact_uri.exists()
    assert artifact.feature_schema_uri.exists()


def test_absolute_ranker_trains_on_v2_absolute_label_and_records_rank_metrics(tmp_path):
    bars = normalized_bars()
    feature_mart = build_feature_mart(str(create_vpa_db(tmp_path / "vpa.duckdb")), bars, "2024-01-02", "2024-01-08", FEATURE_SET_BASELINE_A, [5], build_tradeability_mart(bars))
    samples = build_training_samples(
        feature_mart,
        build_labels(bars, [1], include_v2=True),
        FEATURE_SET_BASELINE_A,
        1,
        "from_next_open",
        label_name="absolute_label",
    )

    artifact = train_alpha_ranker(samples, FEATURE_SET_BASELINE_A, "absolute_label", "from_next_open", 1, tmp_path)

    assert artifact.label_name == "absolute_label"
    assert "train_rank_ic" in artifact.metrics
    assert artifact.metrics["eval_at_10"] == 10.0
    assert artifact.metrics["eval_at_15"] == 15.0
