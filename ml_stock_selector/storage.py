from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd


def init_ml_db(path: Path | str) -> duckdb.DuckDBPyConnection:
    db_path = Path(path)
    if str(path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    schema_path = Path(__file__).resolve().parents[1] / "sql" / "create_ml_tables.sql"
    con.execute(schema_path.read_text(encoding="utf-8"))
    _apply_migrations(con)
    return con


def _apply_migrations(con: duckdb.DuckDBPyConnection) -> None:
    alters = [
        "alter table ml_tradeability_daily add column if not exists is_bse boolean",
        "alter table ml_feature_mart_daily add column if not exists is_bse boolean",
        "alter table ml_predictions_daily add column if not exists run_id varchar",
        "alter table ml_predictions_daily add column if not exists fold_id varchar",
        "alter table ml_predictions_daily add column if not exists absolute_model_id varchar",
        "alter table ml_predictions_daily add column if not exists active_model_id varchar",
        "alter table ml_predictions_daily add column if not exists risk_model_id varchar",
        "alter table ml_model_registry add column if not exists train_start varchar",
        "alter table ml_model_registry add column if not exists train_end varchar",
        "alter table ml_model_registry add column if not exists valid_start varchar",
        "alter table ml_model_registry add column if not exists valid_end varchar",
        "alter table ml_model_registry add column if not exists test_start varchar",
        "alter table ml_model_registry add column if not exists test_end varchar",
        "alter table ml_backtest_metrics add column if not exists fold_id varchar",
        "alter table ml_backtest_metrics add column if not exists score_version varchar",
    ]
    for sql in alters:
        try:
            con.execute(sql)
        except Exception:
            # Keep startup resilient for historical table variants.
            continue


def upsert_dataframe(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
    key_columns: list[str],
) -> None:
    if frame.empty:
        return
    table_columns = [
        row[0]
        for row in con.execute(
            """
            select column_name
            from information_schema.columns
            where table_schema = 'main' and table_name = ?
            order by ordinal_position
            """,
            [table_name],
        ).fetchall()
    ]
    frame = frame[[column for column in table_columns if column in frame.columns]].copy()
    temp_name = f"_ml_upsert_{uuid4().hex}"
    con.register(temp_name, frame)
    condition = " and ".join(
        f"{table_name}.{column} = {temp_name}.{column}" for column in key_columns
    )
    try:
        con.execute(f"delete from {table_name} using {temp_name} where {condition}")
        con.execute(f"insert into {table_name} by name select * from {temp_name}")
    finally:
        con.unregister(temp_name)
