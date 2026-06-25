from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from scripts.import_board_execution_data import import_board_execution_data


def test_import_board_execution_events_csv_and_validate_contract(tmp_path: Path) -> None:
    db = tmp_path / "board.duckdb"
    events = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "first_limit_time": "2024-01-02 10:01:00",
                "last_limit_time": "2024-01-02 14:55:00",
                "seal_duration_seconds": 3600,
                "reopen_count": 1,
                "limit_up": 11.0,
                "close": 11.0,
                "is_close_sealed": True,
            }
        ]
    ).to_csv(events, index=False)

    imported = import_board_execution_data(db_path=db, events_csv=events, source="unit")
    con = duckdb.connect(str(db), read_only=True)
    try:
        row = con.execute("select code, source from board_intraday_events").fetchone()
    finally:
        con.close()

    assert imported == {"board_intraday_events": 1}
    assert row == ("000001.SZ", "unit")


def test_import_board_execution_strict_requires_optional_tables(tmp_path: Path) -> None:
    db = tmp_path / "board.duckdb"
    events = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "first_limit_time": "2024-01-02 10:01:00",
                "last_limit_time": "2024-01-02 14:55:00",
                "seal_duration_seconds": 3600,
                "reopen_count": 1,
                "limit_up": 11.0,
                "close": 11.0,
                "is_close_sealed": True,
            }
        ]
    ).to_csv(events, index=False)

    with pytest.raises(ValueError, match="--require-order-book requires --order-book-csv"):
        import_board_execution_data(
            db_path=db,
            events_csv=events,
            require_order_book=True,
            require_fills=True,
        )


def test_import_board_execution_full_contract(tmp_path: Path) -> None:
    db = tmp_path / "board.duckdb"
    events = tmp_path / "events.csv"
    order_book = tmp_path / "order_book.csv"
    fills = tmp_path / "fills.csv"
    pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "first_limit_time": "2024-01-02 10:01:00",
                "last_limit_time": "2024-01-02 14:55:00",
                "seal_duration_seconds": 3600,
                "reopen_count": 1,
                "limit_up": 11.0,
                "close": 11.0,
                "is_close_sealed": True,
            }
        ]
    ).to_csv(events, index=False)
    pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "snapshot_time": "2024-01-02 10:01:00",
                "bid_price_1": 11.0,
                "bid_volume_1": 1000000,
                "ask_price_1": 0.0,
                "ask_volume_1": 0.0,
                "limit_queue_volume": 1000000,
            }
        ]
    ).to_csv(order_book, index=False)
    pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "signal_time": "2024-01-02 10:01:00",
                "order_time": "2024-01-02 10:01:01",
                "side": "buy",
                "order_price": 11.0,
                "order_qty": 1000,
                "filled_qty": 500,
                "avg_fill_price": 11.0,
                "status": "partial",
            }
        ]
    ).to_csv(fills, index=False)

    imported = import_board_execution_data(
        db_path=db,
        events_csv=events,
        order_book_csv=order_book,
        fills_csv=fills,
        require_order_book=True,
        require_fills=True,
    )

    assert imported == {
        "board_intraday_events": 1,
        "board_order_book_snapshots": 1,
        "board_order_fills": 1,
    }
