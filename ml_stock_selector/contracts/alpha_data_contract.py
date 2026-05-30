from __future__ import annotations

from dataclasses import dataclass

import duckdb


REQUIRED_NORMALIZED_BAR_COLUMNS = {
    "trade_date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "amount",
    "turnover_rate",
    "is_st",
    "is_paused",
    "limit_up",
    "limit_down",
    "industry_code",
}
OPTIONAL_NORMALIZED_BAR_COLUMNS = {"industry_name"}


@dataclass(frozen=True)
class AlphaDataContractResult:
    ok: bool
    missing_tables: list[str]
    missing_columns: dict[str, list[str]]
    warnings: list[str]


def validate_alpha_data_contract(
    con: duckdb.DuckDBPyConnection,
    normalized_table: str = "stock_bar_normalized_daily",
) -> AlphaDataContractResult:
    tables = _tables(con)
    if normalized_table not in tables:
        return AlphaDataContractResult(False, [normalized_table], {}, [])
    columns = _columns(con, normalized_table)
    missing = sorted(REQUIRED_NORMALIZED_BAR_COLUMNS - columns)
    warnings = []
    if "industry_code" in columns and "industry_name" not in columns:
        warnings.append("industry_name missing; reports will fall back to industry_code")
    return AlphaDataContractResult(
        ok=not missing,
        missing_tables=[],
        missing_columns={normalized_table: missing} if missing else {},
        warnings=warnings,
    )


def assert_alpha_data_contract(
    con: duckdb.DuckDBPyConnection,
    normalized_table: str = "stock_bar_normalized_daily",
) -> None:
    result = validate_alpha_data_contract(con, normalized_table)
    if result.ok:
        return
    messages = []
    if result.missing_tables:
        messages.append(f"Missing tables: {', '.join(result.missing_tables)}")
    for table, columns in result.missing_columns.items():
        messages.append(f"Missing columns in {table}: {', '.join(columns)}")
    raise ValueError("; ".join(messages))


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

