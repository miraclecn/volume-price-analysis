from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import duckdb


REQUIRED_VPA_TABLES = {
    "vpa_features": {"date", "scope_type", "scope_id", "window_n", "ret_pct", "vol_rvol_n"},
    "vpa_bar_context_labels": {"date", "scope_type", "scope_id", "window_n", "raw_label"},
    "vpa_sequence_stats": {"date", "scope_type", "scope_id", "window_n"},
    "vpa_structure_state": {"date", "scope_type", "scope_id"},
}


@dataclass(frozen=True)
class VPASchemaContractResult:
    ok: bool
    missing_tables: list[str]
    missing_columns: dict[str, list[str]]


def validate_vpa_schema_contract(con: duckdb.DuckDBPyConnection) -> VPASchemaContractResult:
    tables = _tables(con)
    missing_tables = sorted(set(REQUIRED_VPA_TABLES) - tables)
    missing_columns: dict[str, list[str]] = {}
    for table, required in REQUIRED_VPA_TABLES.items():
        if table not in tables:
            continue
        columns = _columns(con, table)
        missing = sorted(required - columns)
        if missing:
            missing_columns[table] = missing
    return VPASchemaContractResult(not missing_tables and not missing_columns, missing_tables, missing_columns)


def assert_vpa_schema_contract(con: duckdb.DuckDBPyConnection) -> None:
    result = validate_vpa_schema_contract(con)
    if result.ok:
        return
    messages = []
    if result.missing_tables:
        messages.append(f"Missing VPA tables: {', '.join(result.missing_tables)}")
    for table, columns in result.missing_columns.items():
        messages.append(f"Missing columns in {table}: {', '.join(columns)}")
    raise ValueError("; ".join(messages))


def write_vpa_schema_snapshot(con: duckdb.DuckDBPyConnection, path: Path | str) -> None:
    snapshot = {}
    for table in sorted(_tables(con)):
        if table.startswith("vpa_"):
            snapshot[table] = sorted(_columns(con, table))
    Path(path).write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")


def _tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    return {
        row[0]
        for row in con.execute(
            "select table_name from information_schema.tables where table_schema = 'main'"
        ).fetchall()
    }


def _columns(con: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
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

