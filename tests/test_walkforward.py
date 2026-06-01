from __future__ import annotations

from ml_stock_selector.backtest.walkforward import run_walkforward_experiment
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.constants import FEATURE_SET_BASELINE_A
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.tradeability import build_tradeability_mart
from tests.ml_fixtures import create_vpa_db, normalized_bars


def _walkforward_registry_con():
    import duckdb

    con = duckdb.connect(":memory:")
    con.execute(
        """
        create table if not exists ml_model_registry (
            model_id varchar,
            model_type varchar,
                feature_set_id varchar,
                label_name varchar,
                label_base varchar,
                horizon_d integer,
                train_start varchar,
                train_end varchar,
                valid_start varchar,
                valid_end varchar,
                test_start varchar,
                test_end varchar,
                params_json varchar,
                metrics_json varchar,
            feature_schema_uri varchar,
            artifact_uri varchar,
            is_active boolean,
            activated_at varchar,
            deactivated_at varchar,
            created_at varchar,
            notes varchar,
            primary key (model_id)
        )
        """
    )
    return con


def _synthetic_walkforward_inputs(tmp_path):
    bars = normalized_bars()
    tradeability = build_tradeability_mart(bars)
    feature_mart = build_feature_mart(
        str(create_vpa_db(tmp_path / "vpa.duckdb")),
        bars,
        "2024-01-02",
        "2024-01-08",
        FEATURE_SET_BASELINE_A,
        [5],
        tradeability,
    )
    labels = build_labels(bars, [1], include_v2=True)
    config = load_ml_config("config/ml_default.toml")
    config.split["folds"] = [
        {
            "fold_id": "wf_test",
            "train_start": "2024-01-02",
            "train_end": "2024-01-04",
            "valid_start": "2024-01-05",
            "valid_end": "2024-01-05",
            "test_start": "2024-01-05",
            "test_end": "2024-01-08",
        }
    ]
    config.portfolio["min_adv20_amount"] = 0
    config.portfolio["min_trade_score"] = 0.0
    config.ml_v2["candidate_min_count"] = 1
    config.ml_v2["candidate_absolute_min_rank_pct"] = 0.0
    config.ml_v2["candidate_active_min_rank_pct"] = 0.0
    config.ml_v2["candidate_risk_max_rank_pct"] = 1.0
    config.ml_v2["core_absolute_min_rank_pct"] = 0.0
    config.ml_v2["core_active_min_rank_pct"] = 0.0
    config.ml_v2["core_risk_max_rank_pct"] = 1.0
    config.ml_v2["core_min_trade_score"] = -999.0
    return bars, tradeability, feature_mart, labels, config


def test_walkforward_runs_synthetic_fold(tmp_path):
    bars, tradeability, feature_mart, labels, config = _synthetic_walkforward_inputs(tmp_path)
    con = _walkforward_registry_con()
    results = run_walkforward_experiment(config, con, bars, feature_mart, labels, tradeability, artifact_dir=tmp_path)
    con.close()

    assert results
    assert not results[0].predictions.empty
    assert not results[0].targets.empty
    assert not results[0].backtest_result.nav.empty


def test_walkforward_uses_portfolio_v2_core_thresholds(tmp_path):
    bars, tradeability, feature_mart, labels, config = _synthetic_walkforward_inputs(tmp_path)
    config.ml_v2["core_absolute_min_rank_pct"] = 1.01
    con = _walkforward_registry_con()

    results = run_walkforward_experiment(config, con, bars, feature_mart, labels, tradeability, artifact_dir=tmp_path)
    con.close()

    assert results
    assert results[0].targets.empty
