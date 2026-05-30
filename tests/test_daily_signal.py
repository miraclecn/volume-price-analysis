from __future__ import annotations

import duckdb

from ml_stock_selector.constants import FEATURE_SET_BASELINE_A
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.models.alpha_ranker import train_alpha_ranker
from ml_stock_selector.registry import activate_model, register_model
from ml_stock_selector.sample_builder import build_training_samples
from ml_stock_selector.serving.artifact_loader import load_active_model
from ml_stock_selector.serving.daily_signal import generate_daily_signal
from ml_stock_selector.storage import init_ml_db, upsert_dataframe
from ml_stock_selector.tradeability import build_tradeability_mart
from tests.ml_fixtures import create_vpa_db, normalized_bars


def test_daily_signal_loads_active_model_and_writes_predictions(tmp_path):
    bars = normalized_bars()
    tradeability = build_tradeability_mart(bars)
    feature_mart = build_feature_mart(str(create_vpa_db(tmp_path / "vpa.duckdb")), bars, "2024-01-02", "2024-01-08", FEATURE_SET_BASELINE_A, [5], tradeability)
    labels = build_labels(bars, [1])
    samples = build_training_samples(feature_mart, labels, FEATURE_SET_BASELINE_A, 1, "from_next_open")
    artifact = train_alpha_ranker(samples, FEATURE_SET_BASELINE_A, "rank_label", "from_next_open", 1, tmp_path)

    con = init_ml_db(tmp_path / "ml.duckdb")
    upsert_dataframe(con, "ml_feature_mart_daily", feature_mart, ["trade_date", "code", "feature_set_id"])
    register_model(con, model_id=artifact.model_id, model_type=artifact.model_type, feature_set_id=artifact.feature_set_id, label_name=artifact.label_name, label_base=artifact.label_base, horizon_d=artifact.horizon_d, artifact_uri=str(artifact.artifact_uri), feature_schema_uri=str(artifact.feature_schema_uri))
    activate_model(con, artifact.model_id)

    loaded = load_active_model(con, artifact.model_type, FEATURE_SET_BASELINE_A, "rank_label", "from_next_open", 1)
    predictions, targets = generate_daily_signal(con, "2024-01-03", FEATURE_SET_BASELINE_A, 1, "p1")

    assert loaded.model_id == artifact.model_id
    assert not predictions.empty
    assert len(targets) <= 15
    con.close()

