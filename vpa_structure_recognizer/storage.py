from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd


def init_vpa_db(path: Path | str) -> duckdb.DuckDBPyConnection:
    db_path = Path(path)
    if str(path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    schema_path = Path(__file__).resolve().parents[1] / "sql" / "create_vpa_tables.sql"
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

    temp_name = f"_vpa_upsert_{uuid4().hex}"
    con.register(temp_name, frame)
    condition = " and ".join(
        f"{table_name}.{column} = {temp_name}.{column}" for column in key_columns
    )
    try:
        con.execute(f"delete from {table_name} using {temp_name} where {condition}")
        con.execute(f"insert into {table_name} by name select * from {temp_name}")
    finally:
        con.unregister(temp_name)
