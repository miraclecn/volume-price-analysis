from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_stock_selector.contracts.board_execution_contract import (
    REQUIRED_EVENT_COLUMNS,
    REQUIRED_FILL_COLUMNS,
    REQUIRED_ORDER_BOOK_COLUMNS,
)
from scripts.export_board_execution_templates import export_board_execution_templates


def test_export_board_execution_templates_match_contract_columns(tmp_path: Path) -> None:
    written = export_board_execution_templates(tmp_path)

    events = pd.read_csv(tmp_path / "board_intraday_events_template.csv")
    order_book = pd.read_csv(tmp_path / "board_order_book_snapshots_template.csv")
    fills = pd.read_csv(tmp_path / "board_order_fills_template.csv")

    assert set(events.columns) == REQUIRED_EVENT_COLUMNS
    assert set(order_book.columns) == REQUIRED_ORDER_BOOK_COLUMNS
    assert set(fills.columns) == REQUIRED_FILL_COLUMNS
    assert len(written) == 4


def test_export_board_execution_templates_readme_documents_strict_import(tmp_path: Path) -> None:
    export_board_execution_templates(tmp_path)

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert "scripts/import_board_execution_data.py" in readme
    assert "--require-order-book" in readme
    assert "--require-fills" in readme
    assert "status = live_candidate" in readme
