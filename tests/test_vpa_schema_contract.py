from __future__ import annotations

import json

import duckdb
import pytest

from ml_stock_selector.contracts.vpa_schema import (
    assert_vpa_schema_contract,
    validate_vpa_schema_contract,
    write_vpa_schema_snapshot,
)
from tests.ml_fixtures import create_vpa_db


def test_vpa_schema_contract_passes_for_current_vpa_tables(tmp_path):
    db_path = create_vpa_db(tmp_path / "vpa.duckdb")
    con = duckdb.connect(str(db_path))

    result = validate_vpa_schema_contract(con)

    con.close()
    assert result.ok


def test_vpa_schema_contract_errors_for_missing_table(tmp_path):
    con = duckdb.connect(str(tmp_path / "empty.duckdb"))

    with pytest.raises(ValueError, match="vpa_features"):
        assert_vpa_schema_contract(con)

    con.close()


def test_vpa_schema_snapshot_writes_valid_json(tmp_path):
    db_path = create_vpa_db(tmp_path / "vpa.duckdb")
    con = duckdb.connect(str(db_path))
    out = tmp_path / "schema.json"

    write_vpa_schema_snapshot(con, out)

    con.close()
    assert "vpa_features" in json.loads(out.read_text(encoding="utf-8"))

