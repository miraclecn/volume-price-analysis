from __future__ import annotations

import duckdb
import pytest

from ml_stock_selector.contracts.board_execution_contract import (
    assert_board_execution_contract,
    validate_board_execution_contract,
)


def test_board_execution_contract_passes_with_minimum_intraday_event_table() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        """
        create table board_intraday_events (
            trade_date varchar,
            code varchar,
            first_limit_time timestamp,
            last_limit_time timestamp,
            seal_duration_seconds double,
            reopen_count integer,
            limit_up double,
            close double,
            is_close_sealed boolean
        )
        """
    )

    result = validate_board_execution_contract(con)

    assert result.ok is True
    assert result.missing_tables == []
    assert result.missing_columns == {}
    assert "order book snapshots missing" in result.warnings[0]
    assert "broker fill logs missing" in result.warnings[1]


def test_board_execution_contract_fails_without_event_table() -> None:
    con = duckdb.connect(":memory:")

    result = validate_board_execution_contract(con)

    assert result.ok is False
    assert result.missing_tables == ["board_intraday_events"]
    with pytest.raises(ValueError, match="Missing tables: board_intraday_events"):
        assert_board_execution_contract(con)


def test_board_execution_contract_reports_missing_event_columns() -> None:
    con = duckdb.connect(":memory:")
    con.execute("create table board_intraday_events (trade_date varchar, code varchar)")

    result = validate_board_execution_contract(con)

    assert result.ok is False
    assert "first_limit_time" in result.missing_columns["board_intraday_events"]
    assert "is_close_sealed" in result.missing_columns["board_intraday_events"]


def test_board_execution_contract_can_require_order_book_and_fills() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        """
        create table board_intraday_events (
            trade_date varchar,
            code varchar,
            first_limit_time timestamp,
            last_limit_time timestamp,
            seal_duration_seconds double,
            reopen_count integer,
            limit_up double,
            close double,
            is_close_sealed boolean
        )
        """
    )

    result = validate_board_execution_contract(con, require_order_book=True, require_fills=True)

    assert result.ok is False
    assert result.missing_tables == ["board_order_book_snapshots", "board_order_fills"]
