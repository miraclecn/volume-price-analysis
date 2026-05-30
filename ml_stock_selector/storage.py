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
    return con


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
