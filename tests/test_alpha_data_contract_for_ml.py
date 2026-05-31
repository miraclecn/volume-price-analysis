from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from ml_stock_selector.contracts.alpha_data_contract import (
    assert_alpha_data_contract,
    validate_alpha_data_contract,
)
from ml_stock_selector.data_access import (
    load_normalized_stock_bars,
    load_optional_industry_benchmark_returns,
    load_optional_market_benchmark_returns,
)
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


def test_optional_benchmark_loaders_return_empty_when_tables_are_absent(tmp_path):
    db_path = create_alpha_data_db(tmp_path / "alpha.duckdb")

    market = load_optional_market_benchmark_returns(str(db_path), "2024-01-02", "2024-01-04")
    industry = load_optional_industry_benchmark_returns(str(db_path), "2024-01-02", "2024-01-04")

    assert market.empty
    assert industry.empty


def test_optional_benchmark_loaders_read_available_tables(tmp_path):
    db_path = create_alpha_data_db(tmp_path / "alpha.duckdb")
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        create table market_benchmark_daily (
            trade_date varchar,
            horizon_d integer,
            label_base varchar,
            market_ret double
        )
        """
    )
    con.execute(
        """
        create table industry_benchmark_daily (
            trade_date varchar,
            industry_code varchar,
            horizon_d integer,
            label_base varchar,
            industry_ret double,
            benchmark_peer_count integer
        )
        """
    )
    con.execute("insert into market_benchmark_daily values ('2024-01-03', 5, 'from_close', 0.01)")
    con.execute("insert into industry_benchmark_daily values ('2024-01-03', 'I1', 5, 'from_close', 0.02, 8)")
    con.close()

    market = load_optional_market_benchmark_returns(str(db_path), "2024-01-02", "2024-01-04")
    industry = load_optional_industry_benchmark_returns(str(db_path), "2024-01-02", "2024-01-04")

    assert market["market_ret"].tolist() == [0.01]
    assert industry["industry_ret"].tolist() == [0.02]
