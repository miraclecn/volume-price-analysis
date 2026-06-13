from __future__ import annotations

from ml_stock_selector.backtest.walkforward import run_walkforward_experiment
from ml_stock_selector.backtest.walkforward import run_walkforward_feature_store_experiment
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.constants import FEATURE_SET_BASELINE_A
from ml_stock_selector.feature_store import export_feature_store
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.storage import init_ml_db, upsert_dataframe
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
    config.portfolio["v2"]["min_adv20_amount"] = 0
    config.portfolio["min_trade_score"] = 0.0
    config.ml_v2["candidate_min_count"] = 1
    config.ml_v2["candidate_absolute_min_rank_pct"] = 0.0
    config.ml_v2["candidate_active_min_rank_pct"] = 0.0
    config.ml_v2["candidate_risk_max_rank_pct"] = 1.0
    config.ml_v2["candidate_min_trade_score"] = 0.0
    config.ml_v2["core_absolute_min_rank_pct"] = 0.0
    config.ml_v2["core_active_min_rank_pct"] = 0.0
    config.ml_v2["core_risk_max_rank_pct"] = 1.0
    config.ml_v2["core_min_trade_score"] = -999.0
    config.portfolio["v2"]["exit"]["sell_score_threshold"] = -1.0
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


def test_walkforward_writes_phase4_run_fold_artifacts(tmp_path):
    bars, tradeability, feature_mart, labels, config = _synthetic_walkforward_inputs(tmp_path)
    con = _walkforward_registry_con()
    run_root = tmp_path / "runs" / "run_artifacts"

    results = run_walkforward_experiment(
        config,
        con,
        bars,
        feature_mart,
        labels,
        tradeability,
        artifact_dir=tmp_path / "flat_artifacts",
        run_id="run_artifacts",
        run_artifact_root=run_root,
    )
    con.close()

    fold_root = run_root / "folds" / "wf_test"
    assert results
    assert (fold_root / "models" / "absolute_ranker" / "model.pkl").exists()
    assert (fold_root / "models" / "active_ranker" / "params.json").exists()
    assert (fold_root / "models" / "risk_model" / "train_metrics.json").exists()
    assert (fold_root / "predictions" / "scored_predictions.parquet").exists()
    assert (fold_root / "portfolio" / "targets.parquet").exists()
    assert (fold_root / "backtest" / "nav.parquet").exists()


def test_walkforward_backfills_from_candidate_pool_when_core_threshold_excludes_core(tmp_path):
    bars, tradeability, feature_mart, labels, config = _synthetic_walkforward_inputs(tmp_path)
    config.ml_v2["core_absolute_min_rank_pct"] = 1.01
    con = _walkforward_registry_con()

    results = run_walkforward_experiment(config, con, bars, feature_mart, labels, tradeability, artifact_dir=tmp_path)
    con.close()

    assert results
    assert not results[0].targets.empty
    assert set(results[0].targets["entry_reason"]) == {"candidate_pool"}
    diagnostics = results[0].backtest_result.portfolio_diagnostics
    assert diagnostics["selected_from_core"].sum() == 0
    assert diagnostics["selected_from_candidate"].sum() > 0


def test_walkforward_feature_store_path_runs_single_fold_without_json_inputs(tmp_path):
    bars, tradeability, feature_mart, labels, config = _synthetic_walkforward_inputs(tmp_path)
    con = init_ml_db(tmp_path / "ml.duckdb")
    upsert_dataframe(con, "ml_tradeability_daily", tradeability, ["trade_date", "code"])
    upsert_dataframe(con, "ml_feature_mart_daily", feature_mart, ["trade_date", "code", "feature_set_id"])
    upsert_dataframe(con, "ml_labels_daily", labels, ["trade_date", "code", "horizon_d", "label_base"])
    export_feature_store(
        con,
        tmp_path / "feature_store",
        "v2",
        FEATURE_SET_BASELINE_A,
        "2024-01-02",
        "2024-01-08",
        chunk_size=2,
    )

    results = run_walkforward_feature_store_experiment(
        config,
        con,
        bars,
        run_id="run",
        feature_store_dir=str(tmp_path / "feature_store"),
        feature_store_version="v2",
        matrix_cache_dir=str(tmp_path / "cache"),
        feature_set_id=FEATURE_SET_BASELINE_A,
        horizon_d=1,
        label_base="from_next_open",
        score_version="v2_three_model",
        fold_id="wf_test",
        artifact_dir=tmp_path / "artifacts",
        batch_size=2,
        prediction_chunk_size=1,
    )

    raw_rows = con.execute("select count(*) from ml_prediction_raw_daily where run_id = 'run'").fetchone()[0]
    registered = con.execute("select count(*), bool_or(coalesce(is_active, false)) from ml_model_registry").fetchone()
    con.close()
    assert len(results) == 1
    assert raw_rows > 0
    assert registered == (3, False)
