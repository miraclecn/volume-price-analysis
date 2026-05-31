from __future__ import annotations

from ml_stock_selector.constants import FEATURE_SET_BASELINE_A, SCORE_VERSION_THREE_MODEL
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.models.active_ranker import train_active_ranker
from ml_stock_selector.models.alpha_ranker import train_alpha_ranker
from ml_stock_selector.models.risk_model import train_risk_model
from ml_stock_selector.prediction import build_prediction_rows, build_three_model_prediction_rows, predict_with_model
from ml_stock_selector.sample_builder import build_training_samples
from ml_stock_selector.tradeability import build_tradeability_mart
from tests.ml_fixtures import create_vpa_db, normalized_bars


def test_prediction_preserves_keys_and_alpha_scores(tmp_path):
    bars = normalized_bars()
    feature_mart = build_feature_mart(str(create_vpa_db(tmp_path / "vpa.duckdb")), bars, "2024-01-02", "2024-01-08", FEATURE_SET_BASELINE_A, [5], build_tradeability_mart(bars))
    samples = build_training_samples(feature_mart, build_labels(bars, [1]), FEATURE_SET_BASELINE_A, 1, "from_next_open")
    artifact = train_alpha_ranker(samples, FEATURE_SET_BASELINE_A, "rank_label", "from_next_open", 1, tmp_path)

    scores = predict_with_model(samples, artifact)
    rows = build_prediction_rows(samples, scores, artifact)

    assert len(rows) == len(samples)
    assert {"trade_date", "code", "alpha_score", "model_id"}.issubset(rows.columns)


def test_three_model_prediction_rows_include_v2_scores(tmp_path):
    bars = normalized_bars()
    feature_mart = build_feature_mart(str(create_vpa_db(tmp_path / "vpa.duckdb")), bars, "2024-01-02", "2024-01-08", FEATURE_SET_BASELINE_A, [5], build_tradeability_mart(bars))
    labels = build_labels(bars, [1], include_v2=True)
    absolute_samples = build_training_samples(feature_mart, labels, FEATURE_SET_BASELINE_A, 1, "from_next_open", label_name="absolute_label")
    active_samples = build_training_samples(feature_mart, labels, FEATURE_SET_BASELINE_A, 1, "from_next_open", label_name="active_label")
    risk_samples = build_training_samples(feature_mart, labels, FEATURE_SET_BASELINE_A, 1, "from_next_open", label_name="risk_label")
    absolute = train_alpha_ranker(absolute_samples, FEATURE_SET_BASELINE_A, "absolute_label", "from_next_open", 1, tmp_path)
    active = train_active_ranker(active_samples, FEATURE_SET_BASELINE_A, "active_label", "from_next_open", 1, tmp_path)
    risk = train_risk_model(risk_samples, FEATURE_SET_BASELINE_A, "risk_label", "from_next_open", 1, tmp_path)

    rows = build_three_model_prediction_rows(
        feature_mart,
        predict_with_model(feature_mart, absolute),
        predict_with_model(feature_mart, active),
        predict_with_model(feature_mart, risk),
        absolute,
        active,
        risk,
    )

    assert len(rows) == len(feature_mart)
    assert rows["score_version"].eq(SCORE_VERSION_THREE_MODEL).all()
    assert rows["absolute_rank_pct"].between(0.0, 1.0).all()
    assert rows["active_rank_pct"].between(0.0, 1.0).all()
    assert rows["risk_prob"].between(0.0, 1.0).all()
    assert rows["risk_rank_pct"].between(0.0, 1.0).all()
    assert {"absolute_zscore", "active_zscore", "risk_zscore"}.issubset(rows.columns)
