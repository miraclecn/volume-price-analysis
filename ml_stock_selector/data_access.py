from __future__ import annotations

import duckdb
import pandas as pd

from ml_stock_selector.contracts.alpha_data_contract import assert_alpha_data_contract


NORMALIZED_BAR_COLUMNS = [
    "trade_date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "amount",
    "turnover_rate",
    "is_st",
    "is_paused",
    "limit_up",
    "limit_down",
    "industry_code",
    "industry_name",
]


def load_normalized_stock_bars(
    alpha_data_db_path: str,
    start_date: str,
    end_date: str,
    table_name: str = "stock_bar_normalized_daily",
) -> pd.DataFrame:
    con = duckdb.connect(alpha_data_db_path, read_only=True)
    try:
        assert_alpha_data_contract(con, table_name)
        available = {
            row[0]
            for row in con.execute(
                """
                select column_name
                from information_schema.columns
                where table_schema = 'main' and table_name = ?
                """,
                [table_name],
            ).fetchall()
        }
        select_cols = [col for col in NORMALIZED_BAR_COLUMNS if col in available]
        if "industry_name" not in select_cols:
            select_cols.append("cast(null as varchar) as industry_name")
        query = f"""
            select {', '.join(select_cols)}
            from {table_name}
            where trade_date between ? and ?
            order by code, trade_date
        """
        frame = con.execute(query, [start_date, end_date]).fetchdf()
    finally:
        con.close()
    return frame.astype(object).where(pd.notna(frame), None)

