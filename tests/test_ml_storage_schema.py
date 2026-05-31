from __future__ import annotations

import duckdb
import pandas as pd

from ml_stock_selector.storage import init_ml_db, upsert_dataframe


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
