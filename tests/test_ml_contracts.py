from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from ml_stock_selector.backtest.execution import assert_no_t0_fills


def test_vpa_package_does_not_import_ml_stock_selector():
    for path in Path("vpa_structure_recognizer").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                text = ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ""
                assert "ml_stock_selector" not in text


def test_leakage_guard_rejects_same_day_fill():
    orders = pd.DataFrame([{"decision_date": "2024-01-02", "sim_date": "2024-01-02", "status": "filled"}])

    with pytest.raises(ValueError, match="T\\+1"):
        assert_no_t0_fills(orders)

