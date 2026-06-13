from __future__ import annotations

from pathlib import Path

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


def _compact_date(value: str) -> str:
    return value.replace("-", "")


def _normalize_trade_date(value: object) -> object:
    if isinstance(value, str) and len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


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
            where replace(trade_date, '-', '') between ? and ?
            order by code, trade_date
        """
        frame = con.execute(query, [_compact_date(start_date), _compact_date(end_date)]).fetchdf()
    finally:
        con.close()
    frame["trade_date"] = frame["trade_date"].map(_normalize_trade_date)
    return frame.astype(object).where(pd.notna(frame), None)


def load_live_unadjusted_stock_bars(
    alpha_data_db_path: str,
    start_date: str,
    end_date: str,
    normalized_table_name: str = "stock_bar_normalized_daily",
    raw_data_db_path: str | None = None,
    raw_table_name: str = "raw_kline_unadj",
) -> pd.DataFrame:
    bars = load_normalized_stock_bars(alpha_data_db_path, start_date, end_date, normalized_table_name)
    raw_path = Path(raw_data_db_path) if raw_data_db_path is not None else Path(alpha_data_db_path).with_name("raw.duckdb")
    if not raw_path.exists():
        raise FileNotFoundError(f"raw unadjusted bar database not found: {raw_path}")

    raw = _load_raw_unadjusted_bars(str(raw_path), start_date, end_date, raw_table_name)
    if raw.empty:
        raise ValueError(f"no raw unadjusted bars found in {raw_path} for {start_date}..{end_date}")

    merged = bars.merge(raw, on=["trade_date", "code"], how="left", suffixes=("", "_raw"))
    has_raw = merged["close_raw"].notna()
    ratio = pd.to_numeric(merged["close"], errors="coerce") / pd.to_numeric(merged["close_raw"], errors="coerce")
    valid_ratio = has_raw & ratio.notna() & (ratio > 0)

    for column in ["open", "high", "low", "close", "prev_close", "volume", "amount"]:
        raw_column = f"{column}_raw"
        if raw_column in merged:
            merged.loc[has_raw, column] = merged.loc[has_raw, raw_column]

    for column in ["limit_up", "limit_down"]:
        if column in merged:
            merged.loc[valid_ratio, column] = pd.to_numeric(merged.loc[valid_ratio, column], errors="coerce") / ratio.loc[valid_ratio]

    raw_columns = [column for column in merged.columns if column.endswith("_raw")]
    out = merged.drop(columns=raw_columns)
    return out.astype(object).where(pd.notna(out), None)


def _load_raw_unadjusted_bars(
    raw_data_db_path: str,
    start_date: str,
    end_date: str,
    table_name: str,
) -> pd.DataFrame:
    con = duckdb.connect(raw_data_db_path, read_only=True)
    try:
        exists = con.execute(
            """
            select count(*)
            from information_schema.tables
            where table_schema = 'main' and table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
        if not exists:
            raise ValueError(f"raw unadjusted bar table not found: {table_name}")
        frame = con.execute(
            f"""
            select
                trade_date,
                ts_code as code,
                open,
                high,
                low,
                close,
                pre_close as prev_close,
                vol * 100.0 as volume,
                amount * 1000.0 as amount
            from {table_name}
            where replace(cast(trade_date as varchar), '-', '') between ? and ?
            order by code, trade_date
            """,
            [_compact_date(start_date), _compact_date(end_date)],
        ).fetchdf()
    finally:
        con.close()
    if frame.empty:
        return frame
    frame["trade_date"] = frame["trade_date"].map(_normalize_trade_date)
    return frame.astype(object).where(pd.notna(frame), None)


def load_optional_market_benchmark_returns(
    alpha_data_db_path: str,
    start_date: str,
    end_date: str,
    table_name: str = "market_benchmark_daily",
) -> pd.DataFrame:
    return _load_optional_benchmark_table(alpha_data_db_path, table_name, start_date, end_date)


def load_optional_industry_benchmark_returns(
    alpha_data_db_path: str,
    start_date: str,
    end_date: str,
    table_name: str = "industry_benchmark_daily",
) -> pd.DataFrame:
    return _load_optional_benchmark_table(alpha_data_db_path, table_name, start_date, end_date)


def _load_optional_benchmark_table(
    alpha_data_db_path: str,
    table_name: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    con = duckdb.connect(alpha_data_db_path, read_only=True)
    try:
        exists = con.execute(
            """
            select count(*)
            from information_schema.tables
            where table_schema = 'main' and table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
        if not exists:
            return pd.DataFrame()
        frame = con.execute(
            f"""
            select *
            from {table_name}
            where trade_date between ? and ?
            order by trade_date
            """,
            [start_date, end_date],
        ).fetchdf()
    finally:
        con.close()
    return frame.astype(object).where(pd.notna(frame), None)
