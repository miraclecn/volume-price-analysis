from __future__ import annotations

from pathlib import Path

import duckdb

from scripts.research_intraday_data_audit import audit_duckdb_paths


def test_intraday_audit_does_not_treat_scheduler_interval_as_market_data(tmp_path: Path) -> None:
    db = tmp_path / "audit.duckdb"
    con = duckdb.connect(str(db))
    try:
        con.execute("create table mining_schedules (interval_minutes integer)")
        con.execute("create table stock_bar_normalized_daily (trade_date varchar, code varchar, limit_up double, limit_down double)")
    finally:
        con.close()

    report = audit_duckdb_paths([db])

    assert report["summary"]["has_intraday_execution_data"] is False
    assert report["summary"]["daily_available_but_not_enough"] is True
    assert report["summary"]["daily_limit_hit_count"] == 1


def test_intraday_audit_detects_level2_like_fields(tmp_path: Path) -> None:
    db = tmp_path / "audit.duckdb"
    con = duckdb.connect(str(db))
    try:
        con.execute("create table stock_level2_tick (trade_date varchar, code varchar, bid_price_1 double, ask_price_1 double)")
    finally:
        con.close()

    report = audit_duckdb_paths([db])

    assert report["summary"]["has_intraday_execution_data"] is True
    assert report["summary"]["intraday_hit_count"] == 1
