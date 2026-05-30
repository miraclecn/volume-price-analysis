from __future__ import annotations

from ml_stock_selector.backtest.walkforward import run_walkforward_experiment
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.constants import FEATURE_SET_BASELINE_A
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.tradeability import build_tradeability_mart
from tests.ml_fixtures import create_vpa_db, normalized_bars


def test_walkforward_runs_synthetic_fold(tmp_path):
    bars = normalized_bars()
    tradeability = build_tradeability_mart(bars)
    feature_mart = build_feature_mart(str(create_vpa_db(tmp_path / "vpa.duckdb")), bars, "2024-01-02", "2024-01-08", FEATURE_SET_BASELINE_A, [5], tradeability)
    labels = build_labels(bars, [1])
    config = load_ml_config("config/ml_default.toml")

    results = run_walkforward_experiment(config, bars, feature_mart, labels, tradeability, artifact_dir=tmp_path)

    assert results
    assert not results[0].predictions.empty
    assert not results[0].targets.empty
    assert not results[0].backtest_result.nav.empty

