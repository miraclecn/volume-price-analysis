from __future__ import annotations

import duckdb
import pandas as pd

from ml_stock_selector.storage import clear_backtest_outputs, clear_portfolio_targets, init_ml_db, upsert_dataframe


def test_init_ml_db_creates_required_ml_tables(tmp_path):
    db_path = tmp_path / "ml.duckdb"
    con = init_ml_db(db_path)
    con.close()

    check = duckdb.connect(str(db_path))
    tables = {
        row[0]
        for row in check.execute(
            "select table_name from information_schema.tables where table_schema = 'main'"
        ).fetchall()
    }
    check.close()

    assert {
        "ml_tradeability_daily",
        "ml_feature_mart_daily",
        "ml_labels_daily",
        "ml_market_benchmark_daily",
        "ml_industry_benchmark_daily",
        "ml_model_registry",
        "ml_predictions_daily",
        "ml_portfolio_targets_daily",
        "ml_backtest_orders",
        "ml_backtest_positions",
        "ml_backtest_nav",
        "ml_backtest_metrics",
        "ml_runs",
        "ml_run_folds",
    }.issubset(tables)


def test_run_tables_store_run_and_fold_metadata(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    run = pd.DataFrame(
        [
            {
                "run_id": "run_a",
                "run_type": "walkforward",
                "experiment_name": "expanding_gap",
                "config_path": "config/ml_walkforward.toml",
                "config_hash": "hash-a",
                "git_commit": "abc123",
                "alpha_data_db": "/data/research_source.duckdb",
                "ml_db": str(tmp_path / "ml.duckdb"),
                "feature_set_id": "vpa_d_sequence",
                "feature_store_version": "v2",
                "label_version": "from_next_open_h5",
                "score_version": "v2_three_model",
                "artifact_root": str(tmp_path / "runs" / "run_a"),
                "created_at": "t",
                "status": "created",
            }
        ]
    )
    fold = pd.DataFrame(
        [
            {
                "run_id": "run_a",
                "fold_id": "wf_2020",
                "train_start": "2015-01-01",
                "train_end": "2019-12-31",
                "valid_start": "2020-01-01",
                "valid_end": "2020-06-30",
                "test_start": "2020-07-01",
                "test_end": "2020-12-31",
                "gap_type": "one_year_gap",
                "embargo_days": 0,
                "status": "created",
                "artifact_dir": str(tmp_path / "runs" / "run_a" / "folds" / "wf_2020"),
                "created_at": "t",
            }
        ]
    )

    upsert_dataframe(con, "ml_runs", run, ["run_id"])
    upsert_dataframe(con, "ml_run_folds", fold, ["run_id", "fold_id"])

    row = con.execute(
        """
        select r.config_hash, f.test_start, f.artifact_dir
        from ml_runs r
        join ml_run_folds f using (run_id)
        where r.run_id = 'run_a' and f.fold_id = 'wf_2020'
        """
    ).fetchone()
    con.close()

    assert row == ("hash-a", "2020-07-01", str(tmp_path / "runs" / "run_a" / "folds" / "wf_2020"))


def test_ml_schema_exposes_nullable_v2_label_and_prediction_columns(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")

    label_columns = _columns(con, "ml_labels_daily")
    prediction_columns = _columns(con, "ml_predictions_daily")
    con.close()

    assert {
        "absolute_ret",
        "absolute_rank_pct",
        "absolute_label",
        "market_ret",
        "industry_ret",
        "market_excess_ret",
        "industry_excess_ret",
        "active_score",
        "active_rank_pct",
        "active_label",
        "benchmark_missing_market",
        "benchmark_missing_industry",
        "benchmark_peer_count",
    }.issubset(label_columns)
    assert {
        "absolute_score",
        "absolute_rank_pct",
        "absolute_zscore",
        "active_score",
        "active_rank_pct",
        "active_zscore",
        "risk_prob",
        "risk_zscore",
        "core_score",
        "trade_score_v2",
        "score_version",
    }.issubset(prediction_columns)


def test_legacy_label_and_prediction_upserts_still_work_with_v2_schema(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    labels = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "horizon_d": 5,
                "label_base": "from_next_open",
                "base_price": 10.0,
                "future_ret": 0.05,
                "future_score": 0.06,
                "future_rank_pct": 1.0,
                "rank_label": 4,
                "risk_label": 0,
                "outperform_market": True,
                "generated_at": "t",
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "model_id": "legacy_ranker",
                "horizon_d": 5,
                "alpha_score": 0.8,
                "alpha_rank_pct": 1.0,
                "risk_score": 0.1,
                "risk_rank_pct": 0.2,
                "trade_score": 0.9,
                "feature_set_id": "vpa_d_sequence",
                "generated_at": "t",
            }
        ]
    )

    upsert_dataframe(con, "ml_labels_daily", labels, ["trade_date", "code", "horizon_d", "label_base"])
    upsert_dataframe(con, "ml_predictions_daily", predictions, ["trade_date", "code", "model_id", "horizon_d"])

    row = con.execute(
        "select future_ret, active_label from ml_labels_daily where code = '000001.SZ'"
    ).fetchone()
    pred = con.execute(
        "select alpha_score, trade_score_v2 from ml_predictions_daily where code = '000001.SZ'"
    ).fetchone()
    con.close()
    assert row == (0.05, None)
    assert pred == (0.8, None)


def test_ml_upsert_is_idempotent(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    first = pd.DataFrame(
        [{"trade_date": "2024-01-02", "code": "000001.SZ", "adv20_amount": 10.0, "generated_at": "t"}]
    )
    second = first.copy()
    second.loc[0, "adv20_amount"] = 12.0

    upsert_dataframe(con, "ml_tradeability_daily", first, ["trade_date", "code"])
    upsert_dataframe(con, "ml_tradeability_daily", second, ["trade_date", "code"])

    row = con.execute("select count(*), max(adv20_amount) from ml_tradeability_daily").fetchone()
    con.close()
    assert row == (1, 12.0)


def test_backtest_metrics_schema_allows_multiple_folds_per_run(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    metrics = pd.DataFrame(
        [
            {
                "run_id": "run",
                "fold_id": "wf_2020",
                "score_version": "v2",
                "metric_name": "annualized_return",
                "metric_value": 0.1,
                "segment": "fold",
            },
            {
                "run_id": "run",
                "fold_id": "wf_2021",
                "score_version": "v2",
                "metric_name": "annualized_return",
                "metric_value": 0.2,
                "segment": "fold",
            },
        ]
    )

    upsert_dataframe(con, "ml_backtest_metrics", metrics, ["run_id", "fold_id", "score_version", "metric_name", "segment"])

    row = con.execute(
        "select count(*), sum(metric_value) from ml_backtest_metrics where run_id = 'run'"
    ).fetchone()
    con.close()
    assert row == (2, 0.30000000000000004)


def test_backtest_outputs_are_strategy_and_score_scoped_and_clearable(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")

    for table_name in ["ml_backtest_orders", "ml_backtest_positions", "ml_backtest_nav"]:
        assert "fold_id" in _columns(con, table_name)
        assert "strategy_id" in _columns(con, table_name)
        assert "score_version" in _columns(con, table_name)
    assert "order_seq" in _columns(con, "ml_backtest_orders")

    orders = pd.DataFrame(
        [
            {"run_id": "run", "fold_id": "wf_2020", "strategy_id": "s1", "score_version": "v1", "sim_date": "2020-01-03", "decision_date": "2020-01-02", "code": "a", "side": "buy", "order_seq": 1, "status": "filled"},
            {"run_id": "run", "fold_id": "wf_2020", "strategy_id": "s2", "score_version": "v1", "sim_date": "2020-01-03", "decision_date": "2020-01-02", "code": "b", "side": "buy", "order_seq": 1, "status": "filled"},
            {"run_id": "run", "fold_id": "wf_2020", "strategy_id": "s1", "score_version": "v2", "sim_date": "2020-01-03", "decision_date": "2020-01-02", "code": "c", "side": "buy", "order_seq": 1, "status": "filled"},
            {"run_id": "run", "sim_date": "2020-01-04", "decision_date": "2020-01-03", "code": "legacy", "side": "sell", "status": "filled"},
        ]
    )
    positions = pd.DataFrame(
        [
            {"run_id": "run", "fold_id": "wf_2020", "strategy_id": "s1", "score_version": "v1", "sim_date": "2020-01-03", "code": "a"},
            {"run_id": "run", "fold_id": "wf_2020", "strategy_id": "s2", "score_version": "v1", "sim_date": "2020-01-03", "code": "b"},
            {"run_id": "run", "fold_id": "wf_2020", "strategy_id": "s1", "score_version": "v2", "sim_date": "2020-01-03", "code": "c"},
            {"run_id": "run", "sim_date": "2020-01-04", "code": "legacy"},
        ]
    )
    nav = pd.DataFrame(
        [
            {"run_id": "run", "fold_id": "wf_2020", "strategy_id": "s1", "score_version": "v1", "sim_date": "2020-01-03", "nav": 1.0, "cash": 1.0, "gross_exposure": 0.0, "turnover": 0.0},
            {"run_id": "run", "fold_id": "wf_2020", "strategy_id": "s2", "score_version": "v1", "sim_date": "2020-01-03", "nav": 2.0, "cash": 2.0, "gross_exposure": 0.0, "turnover": 0.0},
            {"run_id": "run", "fold_id": "wf_2020", "strategy_id": "s1", "score_version": "v2", "sim_date": "2020-01-03", "nav": 3.0, "cash": 3.0, "gross_exposure": 0.0, "turnover": 0.0},
            {"run_id": "run", "sim_date": "2020-01-04", "nav": 3.0, "cash": 3.0, "gross_exposure": 0.0, "turnover": 0.0},
        ]
    )
    upsert_dataframe(con, "ml_backtest_orders", orders, ["run_id", "fold_id", "strategy_id", "score_version", "sim_date", "decision_date", "code", "side", "order_seq"])
    upsert_dataframe(con, "ml_backtest_positions", positions, ["run_id", "fold_id", "strategy_id", "score_version", "sim_date", "code"])
    upsert_dataframe(con, "ml_backtest_nav", nav, ["run_id", "fold_id", "strategy_id", "score_version", "sim_date"])

    clear_backtest_outputs(con, "run", "wf_2020", "s1", "v1", "2020-01-02", "2020-01-04")

    assert con.execute("select code from ml_backtest_orders order by code").fetchall() == [("b",), ("c",), ("legacy",)]
    assert con.execute("select code from ml_backtest_positions order by code").fetchall() == [("b",), ("c",), ("legacy",)]
    assert con.execute("select strategy_id, score_version, nav from ml_backtest_nav where strategy_id is not null order by nav").fetchall() == [
        ("s2", "v1", 2.0),
        ("s1", "v2", 3.0),
    ]
    con.close()


def test_portfolio_targets_are_run_and_score_scoped(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    assert {"run_id", "fold_id", "score_version"}.issubset(_columns(con, "ml_portfolio_targets_daily"))
    targets = pd.DataFrame(
        [
            {"trade_date": "2020-01-02", "run_id": "run_a", "fold_id": "wf_2020", "portfolio_id": "p", "score_version": "v1", "code": "a"},
            {"trade_date": "2020-01-03", "run_id": "run_b", "fold_id": "wf_2020", "portfolio_id": "p", "score_version": "v1", "code": "b"},
            {"trade_date": "2020-01-03", "run_id": "run_a", "fold_id": "wf_2020", "portfolio_id": "p", "score_version": "v2", "code": "c"},
        ]
    )
    upsert_dataframe(con, "ml_portfolio_targets_daily", targets, ["trade_date", "run_id", "fold_id", "portfolio_id", "score_version", "code"])

    clear_portfolio_targets(con, "run_a", "wf_2020", "p", "v1", "2020-01-01", "2020-01-31")

    rows = con.execute("select run_id, score_version, code from ml_portfolio_targets_daily order by code").fetchall()
    con.close()
    assert rows == [("run_b", "v1", "b"), ("run_a", "v2", "c")]


def _columns(con, table_name: str) -> set[str]:
    return {
        row[0]
        for row in con.execute(
            """
            select column_name
            from information_schema.columns
            where table_schema = 'main' and table_name = ?
            """,
            [table_name],
        ).fetchall()
    }
