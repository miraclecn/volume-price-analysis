from __future__ import annotations

import argparse
from pathlib import Path
import sys

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_stock_selector.contracts.board_execution_contract import (
    assert_board_execution_contract,
    validate_board_execution_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--events-table", default="board_intraday_events")
    parser.add_argument("--order-book-table", default="board_order_book_snapshots")
    parser.add_argument("--fills-table", default="board_order_fills")
    parser.add_argument("--require-order-book", action="store_true")
    parser.add_argument("--require-fills", action="store_true")
    args = parser.parse_args()
    con = duckdb.connect(args.db, read_only=True)
    try:
        result = validate_board_execution_contract(
            con,
            events_table=args.events_table,
            order_book_table=args.order_book_table,
            fills_table=args.fills_table,
            require_order_book=args.require_order_book,
            require_fills=args.require_fills,
        )
        assert_board_execution_contract(
            con,
            events_table=args.events_table,
            order_book_table=args.order_book_table,
            fills_table=args.fills_table,
            require_order_book=args.require_order_book,
            require_fills=args.require_fills,
        )
    finally:
        con.close()
    for warning in result.warnings:
        print(f"warning: {warning}")
    print("board execution contract ok")


if __name__ == "__main__":
    main()
