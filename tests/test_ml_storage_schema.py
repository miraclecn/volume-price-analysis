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
        "ml_model_registry",
        "ml_predictions_daily",
        "ml_portfolio_targets_daily",
        "ml_backtest_orders",
        "ml_backtest_positions",
        "ml_backtest_nav",
        "ml_backtest_metrics",
    }.issubset(tables)


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

