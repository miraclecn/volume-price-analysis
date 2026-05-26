from pathlib import Path

import duckdb

from vpa_structure_recognizer.data_sources import AuditedStockDuckDB, ResearchSourceDuckDB
from vpa_structure_recognizer.models import STOCK_BAR_COLUMNS


def _create_research_source_fixture(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        """
        create table daily_bar_pit (
            security_id varchar,
            trade_date varchar,
            is_st boolean,
            open_adj double,
            high_adj double,
            low_adj double,
            close_adj double,
            pre_close double,
            adj_factor double,
            volume_shares double,
            turnover_value_cny double,
            turnover_rate_pct double
        )
        """
    )
    con.execute(
        """
        create table tradeability_state_daily (
            security_id varchar,
            trade_date varchar,
            is_suspended boolean,
            up_limit double,
            down_limit double
        )
        """
    )
    con.execute(
        """
        create table industry_classification_pit (
            security_id varchar,
            industry_code varchar,
            industry_name varchar,
            effective_at varchar,
            removed_at varchar
        )
        """
    )
    con.executemany(
        "insert into daily_bar_pit values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("000001.SZ", "20240102", False, 10.0, 11.0, 9.8, 10.5, 9.8, 1.0, 1000, 10500, 1.2),
            ("000001.SZ", "20240103", False, 10.5, 11.2, 10.2, 11.0, 10.5, 1.0, 1200, 13200, 1.5),
        ],
    )
    con.executemany(
        "insert into tradeability_state_daily values (?, ?, ?, ?, ?)",
        [
            ("000001.SZ", "20240102", False, 11.55, 9.45),
            ("000001.SZ", "20240103", False, 12.10, 9.90),
        ],
    )
    con.execute(
        "insert into industry_classification_pit values ('000001.SZ', 'BK001', 'Banking', '20200101', null)"
    )
    con.close()


def test_research_source_stock_bars_normalize_pit_adjusted_fields(tmp_path):
    db_path = tmp_path / "research.duckdb"
    _create_research_source_fixture(db_path)

    rows = ResearchSourceDuckDB(db_path).fetch_stock_bars("2024-01-02", "2024-01-03")

    assert list(rows.columns) == STOCK_BAR_COLUMNS
    assert rows["date"].tolist() == ["2024-01-02", "2024-01-03"]
    assert rows["code"].tolist() == ["000001.SZ", "000001.SZ"]
    assert rows["prev_close"].tolist() == [9.8, 10.5]
    assert rows["volume"].tolist() == [1000, 1200]
    assert rows["amount"].tolist() == [10500, 13200]
    assert rows["turnover_rate"].tolist() == [1.2, 1.5]
    assert rows["is_paused"].tolist() == [False, False]
    assert rows["limit_up"].tolist() == [11.55, 12.10]
    assert rows["industry_code"].tolist() == ["BK001", "BK001"]
    assert rows["industry_name"].tolist() == ["Banking", "Banking"]


def test_audited_stock_source_can_extend_history_without_upstream_metadata(tmp_path):
    db_path = tmp_path / "audited.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        create table mart_kline_qfq (
            code varchar,
            date varchar,
            open double,
            high double,
            low double,
            close double,
            vol double,
            amount double,
            pct_chg double
        )
        """
    )
    con.executemany(
        "insert into mart_kline_qfq values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("000001.SZ", "20231229", 9.0, 9.5, 8.8, 9.2, 900, 8280, 1.0),
            ("000001.SZ", "20240102", 9.2, 10.0, 9.1, 9.8, 1100, 10780, 6.5),
        ],
    )
    con.close()

    rows = AuditedStockDuckDB(db_path).fetch_stock_bars("2023-12-29", "2024-01-02")

    assert list(rows.columns) == STOCK_BAR_COLUMNS
    assert rows["prev_close"].tolist() == [None, 9.2]
    assert rows["is_st"].tolist() == [False, False]
    assert rows["industry_code"].isna().all()
