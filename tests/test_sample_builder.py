from __future__ import annotations

from ml_stock_selector.constants import FEATURE_SET_BASELINE_A
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.sample_builder import build_training_samples
from ml_stock_selector.tradeability import build_tradeability_mart
from tests.ml_fixtures import create_vpa_db, normalized_bars


def test_build_training_samples_filters_horizon_and_label_base(tmp_path):
    bars = normalized_bars()
    feature_mart = build_feature_mart(str(create_vpa_db(tmp_path / "vpa.duckdb")), bars, "2024-01-02", "2024-01-08", FEATURE_SET_BASELINE_A, [5], build_tradeability_mart(bars))
    labels = build_labels(bars, [1, 2], label_bases=["from_close", "from_next_open"])

    samples = build_training_samples(feature_mart, labels, FEATURE_SET_BASELINE_A, 2, "from_next_open")

    assert not samples.empty
    assert set(samples["horizon_d"]) == {2}
    assert set(samples["label_base"]) == {"from_next_open"}

