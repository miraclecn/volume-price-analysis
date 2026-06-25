from __future__ import annotations

from dataclasses import dataclass

import duckdb


DEFAULT_EVENTS_TABLE = "board_intraday_events"
DEFAULT_ORDER_BOOK_TABLE = "board_order_book_snapshots"
DEFAULT_FILLS_TABLE = "board_order_fills"

REQUIRED_EVENT_COLUMNS = {
    "trade_date",
    "code",
    "first_limit_time",
    "last_limit_time",
    "seal_duration_seconds",
    "reopen_count",
    "limit_up",
    "close",
    "is_close_sealed",
}

REQUIRED_ORDER_BOOK_COLUMNS = {
    "trade_date",
    "code",
    "snapshot_time",
    "bid_price_1",
    "bid_volume_1",
    "ask_price_1",
    "ask_volume_1",
    "limit_queue_volume",
}

REQUIRED_FILL_COLUMNS = {
    "trade_date",
    "code",
    "signal_time",
    "order_time",
    "side",
    "order_price",
    "order_qty",
    "filled_qty",
    "avg_fill_price",
    "status",
}


@dataclass(frozen=True)
class BoardExecutionContractResult:
    ok: bool
    missing_tables: list[str]
    missing_columns: dict[str, list[str]]
    warnings: list[str]


def validate_board_execution_contract(
    con: duckdb.DuckDBPyConnection,
    *,
    events_table: str = DEFAULT_EVENTS_TABLE,
    order_book_table: str = DEFAULT_ORDER_BOOK_TABLE,
    fills_table: str = DEFAULT_FILLS_TABLE,
    require_order_book: bool = False,
    require_fills: bool = False,
) -> BoardExecutionContractResult:
    tables = _tables(con)
    missing_tables = []
    missing_columns: dict[str, list[str]] = {}
    warnings = []

    if events_table not in tables:
        missing_tables.append(events_table)
    else:
        missing = sorted(REQUIRED_EVENT_COLUMNS - _columns(con, events_table))
        if missing:
            missing_columns[events_table] = missing

    for table_name, required, required_flag, warning in [
        (
            order_book_table,
            REQUIRED_ORDER_BOOK_COLUMNS,
            require_order_book,
            "order book snapshots missing; fillability and queue adverse selection cannot be modeled",
        ),
        (
            fills_table,
            REQUIRED_FILL_COLUMNS,
            require_fills,
            "broker fill logs missing; actual fill rate cannot be calibrated",
        ),
    ]:
        if table_name not in tables:
            if required_flag:
                missing_tables.append(table_name)
            else:
                warnings.append(warning)
            continue
        missing = sorted(required - _columns(con, table_name))
        if missing:
            missing_columns[table_name] = missing

    return BoardExecutionContractResult(
        ok=not missing_tables and not missing_columns,
        missing_tables=sorted(missing_tables),
        missing_columns=missing_columns,
        warnings=warnings,
    )


def assert_board_execution_contract(
    con: duckdb.DuckDBPyConnection,
    *,
    events_table: str = DEFAULT_EVENTS_TABLE,
    order_book_table: str = DEFAULT_ORDER_BOOK_TABLE,
    fills_table: str = DEFAULT_FILLS_TABLE,
    require_order_book: bool = False,
    require_fills: bool = False,
) -> None:
    result = validate_board_execution_contract(
        con,
        events_table=events_table,
        order_book_table=order_book_table,
        fills_table=fills_table,
        require_order_book=require_order_book,
        require_fills=require_fills,
    )
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
