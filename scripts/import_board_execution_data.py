from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_stock_selector.contracts.board_execution_contract import validate_board_execution_contract
from ml_stock_selector.storage import upsert_dataframe


TABLE_KEYS = {
    "board_intraday_events": ["trade_date", "code"],
    "board_order_book_snapshots": ["trade_date", "code", "snapshot_time"],
    "board_order_fills": ["trade_date", "code", "signal_time", "order_time", "side"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--events-csv")
    parser.add_argument("--order-book-csv")
    parser.add_argument("--fills-csv")
    parser.add_argument("--source", default="manual_import")
    parser.add_argument("--require-order-book", action="store_true")
    parser.add_argument("--require-fills", action="store_true")
    args = parser.parse_args()
    imported = import_board_execution_data(
        db_path=Path(args.db),
        events_csv=Path(args.events_csv) if args.events_csv else None,
        order_book_csv=Path(args.order_book_csv) if args.order_book_csv else None,
        fills_csv=Path(args.fills_csv) if args.fills_csv else None,
        source=args.source,
        require_order_book=args.require_order_book,
        require_fills=args.require_fills,
    )
    print(f"imported={imported}")


def import_board_execution_data(
    *,
    db_path: Path,
    events_csv: Path | None = None,
    order_book_csv: Path | None = None,
    fills_csv: Path | None = None,
    source: str = "manual_import",
    require_order_book: bool = False,
    require_fills: bool = False,
) -> dict[str, int]:
    if not any([events_csv, order_book_csv, fills_csv]):
        raise ValueError("at least one CSV path is required")
    if require_order_book and order_book_csv is None:
        raise ValueError("--require-order-book requires --order-book-csv")
    if require_fills and fills_csv is None:
        raise ValueError("--require-fills requires --fills-csv")
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        _init_board_execution_db(con)
        imported = {}
        for table_name, csv_path in [
            ("board_intraday_events", events_csv),
            ("board_order_book_snapshots", order_book_csv),
            ("board_order_fills", fills_csv),
        ]:
            if csv_path is None:
                continue
            frame = _read_import_csv(csv_path, source)
            upsert_dataframe(con, table_name, frame, TABLE_KEYS[table_name])
            imported[table_name] = int(len(frame))
        result = validate_board_execution_contract(
            con,
            require_order_book=require_order_book,
            require_fills=require_fills,
        )
        if not result.ok:
            messages = []
            if result.missing_tables:
                messages.append(f"missing tables: {', '.join(result.missing_tables)}")
            for table, columns in result.missing_columns.items():
                messages.append(f"missing columns in {table}: {', '.join(columns)}")
            raise ValueError("; ".join(messages))
        for warning in result.warnings:
            print(f"warning: {warning}")
        return imported
    finally:
        con.close()


def _init_board_execution_db(con: duckdb.DuckDBPyConnection) -> None:
    schema_path = Path(__file__).resolve().parents[1] / "sql" / "create_board_execution_tables.sql"
    con.execute(schema_path.read_text(encoding="utf-8"))


def _read_import_csv(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    out = frame.copy()
    out["source"] = source
    out["ingested_at"] = datetime.now(timezone.utc).isoformat()
    return out


if __name__ == "__main__":
    main()
