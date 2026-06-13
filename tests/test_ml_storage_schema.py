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
    }.issubset(tables)


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


def test_backtest_outputs_are_fold_scoped_and_clearable(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")

    for table_name in ["ml_backtest_orders", "ml_backtest_positions", "ml_backtest_nav"]:
        assert "fold_id" in _columns(con, table_name)

    orders = pd.DataFrame(
        [
            {"run_id": "run", "fold_id": "wf_2020", "sim_date": "2020-01-03", "decision_date": "2020-01-02", "code": "a", "side": "buy", "status": "filled"},
            {"run_id": "run", "fold_id": "wf_2021", "sim_date": "2020-01-03", "decision_date": "2020-01-02", "code": "b", "side": "buy", "status": "filled"},
            {"run_id": "run", "sim_date": "2020-01-04", "decision_date": "2020-01-03", "code": "legacy", "side": "sell", "status": "filled"},
        ]
    )
    positions = pd.DataFrame(
        [
            {"run_id": "run", "fold_id": "wf_2020", "sim_date": "2020-01-03", "code": "a"},
            {"run_id": "run", "fold_id": "wf_2021", "sim_date": "2020-01-03", "code": "b"},
            {"run_id": "run", "sim_date": "2020-01-04", "code": "legacy"},
        ]
    )
    nav = pd.DataFrame(
        [
            {"run_id": "run", "fold_id": "wf_2020", "sim_date": "2020-01-03", "nav": 1.0, "cash": 1.0, "gross_exposure": 0.0, "turnover": 0.0},
            {"run_id": "run", "fold_id": "wf_2021", "sim_date": "2020-01-03", "nav": 2.0, "cash": 2.0, "gross_exposure": 0.0, "turnover": 0.0},
            {"run_id": "run", "sim_date": "2020-01-04", "nav": 3.0, "cash": 3.0, "gross_exposure": 0.0, "turnover": 0.0},
        ]
    )
    upsert_dataframe(con, "ml_backtest_orders", orders, ["run_id", "fold_id", "sim_date", "decision_date", "code", "side"])
    upsert_dataframe(con, "ml_backtest_positions", positions, ["run_id", "fold_id", "sim_date", "code"])
    upsert_dataframe(con, "ml_backtest_nav", nav, ["run_id", "fold_id", "sim_date"])

    clear_backtest_outputs(con, "run", "wf_2020", "2020-01-02", "2020-01-04")

    assert con.execute("select code from ml_backtest_orders order by code").fetchall() == [("b",)]
    assert con.execute("select code from ml_backtest_positions order by code").fetchall() == [("b",)]
    assert con.execute("select fold_id, nav from ml_backtest_nav").fetchall() == [("wf_2021", 2.0)]
    con.close()


def test_portfolio_targets_are_clearable_by_portfolio_and_date_range(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    targets = pd.DataFrame(
        [
            {"trade_date": "2020-01-02", "portfolio_id": "wf_2020", "code": "a"},
            {"trade_date": "2020-01-03", "portfolio_id": "wf_2020", "code": "b"},
            {"trade_date": "2020-01-03", "portfolio_id": "wf_2021", "code": "c"},
        ]
    )
    upsert_dataframe(con, "ml_portfolio_targets_daily", targets, ["trade_date", "portfolio_id", "code"])

    clear_portfolio_targets(con, "wf_2020", "2020-01-01", "2020-01-31")

    rows = con.execute("select portfolio_id, code from ml_portfolio_targets_daily order by code").fetchall()
    con.close()
    assert rows == [("wf_2021", "c")]


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
