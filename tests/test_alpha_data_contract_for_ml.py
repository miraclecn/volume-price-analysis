from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from ml_stock_selector.contracts.alpha_data_contract import (
    assert_alpha_data_contract,
    validate_alpha_data_contract,
)
from ml_stock_selector.data_access import load_normalized_stock_bars
from tests.ml_fixtures import create_alpha_data_db, normalized_bars


def test_alpha_data_contract_passes_for_normalized_bars(tmp_path):
    db_path = create_alpha_data_db(tmp_path / "alpha.duckdb")
    con = duckdb.connect(str(db_path))

    result = validate_alpha_data_contract(con)

    con.close()
    assert result.ok
    assert result.missing_tables == []


def test_alpha_data_contract_warns_for_missing_industry_name(tmp_path):
    frame = normalized_bars().drop(columns=["industry_name"])
    db_path = create_alpha_data_db(tmp_path / "alpha.duckdb", frame)
    con = duckdb.connect(str(db_path))

    result = validate_alpha_data_contract(con)

    con.close()
    assert result.ok
    assert "industry_name" in result.warnings[0]


def test_alpha_data_contract_fails_for_missing_core_column(tmp_path):
    frame = normalized_bars().drop(columns=["close"])
    db_path = create_alpha_data_db(tmp_path / "alpha.duckdb", frame)
    con = duckdb.connect(str(db_path))

    with pytest.raises(ValueError, match="close"):
        assert_alpha_data_contract(con)

    con.close()


def test_load_normalized_stock_bars_reads_only_contract_table(tmp_path):
    db_path = create_alpha_data_db(tmp_path / "alpha.duckdb")

    frame = load_normalized_stock_bars(str(db_path), "2024-01-03", "2024-01-04")

    assert list(frame[["code", "trade_date"]].iloc[0]) == ["000001.SZ", "2024-01-03"]
    assert set(frame["trade_date"]) == {"2024-01-03", "2024-01-04"}

