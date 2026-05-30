from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from vpa_structure_recognizer.models import STOCK_BAR_COLUMNS


def _compact_date(value: str) -> str:
    return value.replace("-", "")


def _nullable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(object).where(pd.notna(frame), None)


class ResearchSourceDuckDB:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def fetch_stock_bars(self, start_date: str, end_date: str) -> pd.DataFrame:
        start = _compact_date(start_date)
        end = _compact_date(end_date)
        con = duckdb.connect(str(self.path), read_only=True)
        _validate_normalized_contract(con)
        query = """
            select
                case
                    when length(cast(trade_date as varchar)) = 8
                        then substr(cast(trade_date as varchar), 1, 4)
                            || '-' || substr(cast(trade_date as varchar), 5, 2)
                            || '-' || substr(cast(trade_date as varchar), 7, 2)
                    else cast(trade_date as varchar)
                end as date,
                code,
                open,
                high,
                low,
                close,
                prev_close,
                volume,
                amount,
                turnover_rate,
                coalesce(is_st, false) as is_st,
                coalesce(is_paused, false) as is_paused,
                limit_up,
                limit_down,
                industry_code,
                industry_name
            from stock_bar_normalized_daily
            where replace(cast(trade_date as varchar), '-', '') between ? and ?
            order by code, trade_date
        """
        frame = con.execute(query, [start, end]).fetchdf()
        con.close()
        frame = frame.drop_duplicates(["date", "code"], keep="last")
        frame = _nullable_frame(frame[STOCK_BAR_COLUMNS])
        _validate_core_market_fields(frame)
        return frame


class AuditedStockDuckDB:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def fetch_stock_bars(self, start_date: str, end_date: str) -> pd.DataFrame:
        start = _compact_date(start_date)
        end = _compact_date(end_date)
        con = duckdb.connect(str(self.path), read_only=True)
        frame = con.execute(
            """
            with bars as (
                select
                    code,
                    date,
                    open,
                    high,
                    low,
                    close,
                    lag(close) over (
                        partition by code
                        order by date
                    ) as prev_close,
                    vol,
                    amount
                from mart_kline_qfq
            )
            select
                substr(date, 1, 4) || '-' || substr(date, 5, 2) || '-' || substr(date, 7, 2) as date,
                code,
                open,
                high,
                low,
                close,
                prev_close,
                vol as volume,
                amount,
                cast(null as double) as turnover_rate,
                false as is_st,
                false as is_paused,
                cast(null as double) as limit_up,
                cast(null as double) as limit_down,
                cast(null as varchar) as industry_code,
                cast(null as varchar) as industry_name
            from bars
            where date between ? and ?
            order by code, date
            """,
            [start, end],
        ).fetchdf()
        con.close()
        frame = frame.drop_duplicates(["date", "code"], keep="last")
        return _nullable_frame(frame[STOCK_BAR_COLUMNS])


def _validate_normalized_contract(con: duckdb.DuckDBPyConnection) -> None:
    required_columns = set(STOCK_BAR_COLUMNS) - {"date"} | {"trade_date"}
    tables = {
        row[0]
        for row in con.execute(
            "select table_name from information_schema.tables where table_schema = 'main'"
        ).fetchall()
    }
    if "stock_bar_normalized_daily" not in tables:
        raise ValueError("Missing tables: stock_bar_normalized_daily")
    actual_columns = {
        row[0]
        for row in con.execute(
            """
            select column_name
            from information_schema.columns
            where table_schema = 'main' and table_name = 'stock_bar_normalized_daily'
            """
        ).fetchall()
    }
    missing = sorted(required_columns - actual_columns)
    if missing:
        raise ValueError(
            "Missing columns in stock_bar_normalized_daily: " + ", ".join(missing)
        )


def _validate_core_market_fields(frame: pd.DataFrame) -> None:
    core_columns = ["open", "high", "low", "close", "volume", "amount"]
    missing = [column for column in core_columns if frame[column].isna().any()]
    if missing:
        raise ValueError("Invalid null core market fields: " + ", ".join(missing))
