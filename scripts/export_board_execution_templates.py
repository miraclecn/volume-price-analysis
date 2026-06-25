from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_stock_selector.contracts.board_execution_contract import (
    REQUIRED_EVENT_COLUMNS,
    REQUIRED_FILL_COLUMNS,
    REQUIRED_ORDER_BOOK_COLUMNS,
)


DEFAULT_OUT_DIR = "outputs/limit_hit_research/board_execution_templates"

EVENT_COLUMNS = [
    "trade_date",
    "code",
    "first_limit_time",
    "last_limit_time",
    "seal_duration_seconds",
    "reopen_count",
    "limit_up",
    "close",
    "is_close_sealed",
]
ORDER_BOOK_COLUMNS = [
    "trade_date",
    "code",
    "snapshot_time",
    "bid_price_1",
    "bid_volume_1",
    "ask_price_1",
    "ask_volume_1",
    "limit_queue_volume",
]
FILL_COLUMNS = [
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
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    written = export_board_execution_templates(Path(args.out_dir))
    for path in written:
        print(path)


def export_board_execution_templates(out_dir: Path) -> list[Path]:
    _assert_contract_columns()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [
        _write_template(out_dir / "board_intraday_events_template.csv", EVENT_COLUMNS, _event_example()),
        _write_template(out_dir / "board_order_book_snapshots_template.csv", ORDER_BOOK_COLUMNS, _order_book_example()),
        _write_template(out_dir / "board_order_fills_template.csv", FILL_COLUMNS, _fill_example()),
    ]
    readme = out_dir / "README.md"
    readme.write_text(_readme_text(), encoding="utf-8")
    files.append(readme)
    return files


def _write_template(path: Path, columns: list[str], example: dict[str, object]) -> Path:
    pd.DataFrame([example], columns=columns).to_csv(path, index=False)
    return path


def _assert_contract_columns() -> None:
    if set(EVENT_COLUMNS) != REQUIRED_EVENT_COLUMNS:
        raise AssertionError("event template columns diverge from board execution contract")
    if set(ORDER_BOOK_COLUMNS) != REQUIRED_ORDER_BOOK_COLUMNS:
        raise AssertionError("order book template columns diverge from board execution contract")
    if set(FILL_COLUMNS) != REQUIRED_FILL_COLUMNS:
        raise AssertionError("fill template columns diverge from board execution contract")


def _event_example() -> dict[str, object]:
    return {
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


def _order_book_example() -> dict[str, object]:
    return {
        "trade_date": "2024-01-02",
        "code": "000001.SZ",
        "snapshot_time": "2024-01-02 10:01:00",
        "bid_price_1": 11.0,
        "bid_volume_1": 1_000_000,
        "ask_price_1": 0.0,
        "ask_volume_1": 0.0,
        "limit_queue_volume": 1_000_000,
    }


def _fill_example() -> dict[str, object]:
    return {
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


def _readme_text() -> str:
    return """# Board Execution Data Templates

These CSV templates are the required import shape for real board-hitting fillability research.

Import command:

```bash
python scripts/import_board_execution_data.py \\
  --db outputs/limit_hit_research/board_execution.duckdb \\
  --events-csv outputs/limit_hit_research/board_execution_templates/board_intraday_events_template.csv \\
  --order-book-csv outputs/limit_hit_research/board_execution_templates/board_order_book_snapshots_template.csv \\
  --fills-csv outputs/limit_hit_research/board_execution_templates/board_order_fills_template.csv \\
  --require-order-book \\
  --require-fills \\
  --source <vendor_or_broker>
```

Replace the example row with real data before importing.

Table purposes:

- `board_intraday_events_template.csv`: limit-up event timing and seal quality.
- `board_order_book_snapshots_template.csv`: queue/depth snapshots around signal/order time.
- `board_order_fills_template.csv`: actual broker/order fill logs.

After import, run:

```bash
python scripts/run_board_execution_contract_check.py \\
  --db outputs/limit_hit_research/board_execution.duckdb \\
  --require-order-book \\
  --require-fills
```

Do not connect board strategy research to live sim until the candidate promotion manifest reports `status = live_candidate`.
"""


if __name__ == "__main__":
    main()
