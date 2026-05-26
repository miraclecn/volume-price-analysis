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
        industry_name_expr = (
            "industry_name"
            if _has_column(con, "industry_classification_pit", "industry_name")
            else "cast(null as varchar) as industry_name"
        )
        query = f"""
            with bars as (
                select
                    security_id,
                    trade_date,
                    is_st,
                    open_adj,
                    high_adj,
                    low_adj,
                    close_adj,
                    pre_close * adj_factor as prev_close,
                    volume_shares,
                    turnover_value_cny,
                    turnover_rate_pct
                from daily_bar_pit
            ),
            industry as (
                select
                    security_id,
                    industry_code,
                    {industry_name_expr},
                    effective_at,
                    removed_at
                from industry_classification_pit
            )
            select
                substr(b.trade_date, 1, 4) || '-' || substr(b.trade_date, 5, 2) || '-' || substr(b.trade_date, 7, 2) as date,
                b.security_id as code,
                b.open_adj as open,
                b.high_adj as high,
                b.low_adj as low,
                b.close_adj as close,
                b.prev_close,
                b.volume_shares as volume,
                b.turnover_value_cny as amount,
                b.turnover_rate_pct as turnover_rate,
                coalesce(b.is_st, false) as is_st,
                coalesce(t.is_suspended, false) as is_paused,
                t.up_limit as limit_up,
                t.down_limit as limit_down,
                i.industry_code,
                i.industry_name
            from bars b
            left join tradeability_state_daily t
                on t.security_id = b.security_id
                and t.trade_date = b.trade_date
            left join industry i
                on i.security_id = b.security_id
                and b.trade_date >= i.effective_at
                and (i.removed_at is null or b.trade_date < i.removed_at)
            where b.trade_date between ? and ?
            order by b.security_id, b.trade_date
        """
        frame = con.execute(query, [start, end]).fetchdf()
        con.close()
        frame = frame.drop_duplicates(["date", "code"], keep="last")
        return _nullable_frame(frame[STOCK_BAR_COLUMNS])


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


def _has_column(con: duckdb.DuckDBPyConnection, table_name: str, column_name: str) -> bool:
    return bool(
        con.execute(
            """
            select 1
            from information_schema.columns
            where table_schema = 'main'
              and table_name = ?
              and column_name = ?
            """,
            [table_name, column_name],
        ).fetchone()
    )
