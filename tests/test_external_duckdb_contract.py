import os
from pathlib import Path

import duckdb
import pytest

from vpa_structure_recognizer.data_sources import ResearchSourceDuckDB
from vpa_structure_recognizer.models import STOCK_BAR_COLUMNS


SOURCE_DB = Path("/home/nan/alpha-find-v2/output/research_source.duckdb")

pytestmark = pytest.mark.skipif(
    os.getenv("VPA_RUN_EXTERNAL_DUCKDB_TESTS") != "1",
    reason="external DuckDB contract tests are opt-in",
)


def test_external_research_source_has_required_tables_and_columns():
    con = duckdb.connect(str(SOURCE_DB), read_only=True)
    required = {
        "daily_bar_pit": {
            "security_id",
            "trade_date",
            "open_adj",
            "high_adj",
            "low_adj",
            "close_adj",
            "volume_shares",
            "turnover_value_cny",
            "turnover_rate_pct",
            "is_st",
        },
        "tradeability_state_daily": {
            "security_id",
            "trade_date",
            "is_suspended",
            "up_limit",
            "down_limit",
        },
        "industry_classification_pit": {
            "security_id",
            "industry_code",
            "effective_at",
            "removed_at",
        },
    }
    for table, columns in required.items():
        actual = {
            row[0]
            for row in con.execute(
                """
                select column_name
                from information_schema.columns
                where table_name = ?
                """,
                [table],
            ).fetchall()
        }
        assert columns.issubset(actual)
    con.close()


def test_external_research_source_normalizes_small_stock_slice():
    rows = ResearchSourceDuckDB(SOURCE_DB).fetch_stock_bars("2024-01-02", "2024-01-05")

    assert list(rows.columns) == STOCK_BAR_COLUMNS
    assert not rows.empty
    assert rows["date"].str.match(r"\d{4}-\d{2}-\d{2}").all()
