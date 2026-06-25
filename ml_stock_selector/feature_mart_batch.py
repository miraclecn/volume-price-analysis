from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re

import duckdb
import pandas as pd

from ml_stock_selector.data_access import load_normalized_stock_bars
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.storage import init_ml_db, upsert_dataframe
from ml_stock_selector.tradeability import build_tradeability_mart
from vpa_structure_recognizer.batch_runner import BatchPeriod


def feature_mart_period_batches(
    start_date: str,
    end_date: str,
    warmup_months: int = 13,
    batch_months: int = 1,
) -> list[BatchPeriod]:
    if batch_months < 1:
        raise ValueError("batch_months must be positive")

    start = pd.Timestamp(start_date).replace(day=1)
    end = pd.Timestamp(end_date)
    batches: list[BatchPeriod] = []
    period_start = start
    while period_start <= end:
        period_end_month = period_start + pd.DateOffset(months=batch_months - 1)
        period_end = min(period_end_month + pd.offsets.MonthEnd(0), end)
        pre_start = period_start - pd.DateOffset(months=warmup_months)
        batches.append(
            BatchPeriod(
                pre_start.strftime("%Y-%m-%d"),
                period_start.strftime("%Y-%m-%d"),
                period_end.strftime("%Y-%m-%d"),
            )
        )
        period_start = period_start + pd.DateOffset(months=batch_months)
    return batches


def run_feature_mart_batch(
    *,
    alpha_data_db: str,
    vpa_db: str,
    ml_db: str,
    normalized_bars_table: str,
    batch: BatchPeriod,
    feature_set_id: str,
    windows: list[int],
    exclude_industry_metadata_from_features_json: bool,
    lookahead_days: int = 31,
) -> dict[str, int | str]:
    read_end = _add_calendar_days(batch.out_end, lookahead_days)
    bars = load_normalized_stock_bars(
        alpha_data_db,
        batch.pre_start,
        read_end,
        normalized_bars_table,
    )
    tradeability = build_tradeability_mart(bars)
    tradeability_window = _keep_output_window(tradeability, "trade_date", batch)
    feature_mart = build_feature_mart(
        vpa_db,
        bars,
        batch.out_start,
        batch.out_end,
        feature_set_id,
        windows,
        tradeability,
        exclude_industry_metadata_from_features_json=exclude_industry_metadata_from_features_json,
    )

    con = init_ml_db(ml_db)
    try:
        con.execute("pragma threads=1")
        upsert_dataframe(con, "ml_tradeability_daily", tradeability_window, ["trade_date", "code"])
        upsert_dataframe(con, "ml_feature_mart_daily", feature_mart, ["trade_date", "code", "feature_set_id"])
        con.commit()
    finally:
        con.close()

    return {
        "read_start": batch.pre_start,
        "read_end": read_end,
        "out_start": batch.out_start,
        "out_end": batch.out_end,
        "source_rows": len(bars),
        "tradeability": len(tradeability_window),
        "feature_mart": len(feature_mart),
    }


def completed_feature_mart_month_keys(
    *,
    ml_db: str,
    alpha_data_db: str,
    normalized_bars_table: str,
    batches: list[BatchPeriod],
    feature_set_id: str,
) -> set[str]:
    if not batches or not Path(ml_db).exists():
        return set()

    table_name = _safe_identifier(normalized_bars_table)
    out = duckdb.connect(ml_db, read_only=True)
    src = duckdb.connect(alpha_data_db, read_only=True)
    try:
        tables = {
            row[0]
            for row in out.execute(
                "select table_name from information_schema.tables where table_schema = 'main'"
            ).fetchall()
        }
        if "ml_feature_mart_daily" not in tables or "ml_tradeability_daily" not in tables:
            return set()

        start = min(batch.out_start for batch in batches)
        end = max(batch.out_end for batch in batches)
        source_dates = {
            row[0]
            for row in src.execute(
                f"""
                select distinct
                    case
                        when length(cast(trade_date as varchar)) = 8
                            then substr(cast(trade_date as varchar), 1, 4)
                                || '-' || substr(cast(trade_date as varchar), 5, 2)
                                || '-' || substr(cast(trade_date as varchar), 7, 2)
                        else cast(trade_date as varchar)
                    end as date
                from {table_name}
                where replace(cast(trade_date as varchar), '-', '') between ? and ?
                """,
                [start.replace("-", ""), end.replace("-", "")],
            ).fetchall()
        }
        feature_dates = {
            row[0]
            for row in out.execute(
                """
                select distinct trade_date
                from ml_feature_mart_daily
                where feature_set_id = ? and trade_date between ? and ?
                """,
                [feature_set_id, start, end],
            ).fetchall()
        }
        tradeability_dates = {
            row[0]
            for row in out.execute(
                """
                select distinct trade_date
                from ml_tradeability_daily
                where trade_date between ? and ?
                """,
                [start, end],
            ).fetchall()
        }
    finally:
        out.close()
        src.close()

    completed: set[str] = set()
    for batch in batches:
        expected = {date for date in source_dates if batch.out_start <= date <= batch.out_end}
        actual_features = {date for date in feature_dates if batch.out_start <= date <= batch.out_end}
        actual_tradeability = {date for date in tradeability_dates if batch.out_start <= date <= batch.out_end}
        if expected and expected <= actual_features and expected <= actual_tradeability:
            completed.add(batch.month_key)
    return completed


def write_feature_mart_manifest(
    *,
    manifest_path: str,
    ml_db: str,
    alpha_data_db: str,
    vpa_db: str,
    config_path: str,
    start_date: str,
    end_date: str,
    feature_set_id: str,
    windows: list[int],
    warmup_months: int,
    lookahead_days: int,
    batch_months: int,
    planned_batches: int,
) -> dict[str, object]:
    con = duckdb.connect(ml_db, read_only=True)
    try:
        feature_count, feature_min, feature_max = con.execute(
            """
            select count(*), min(trade_date), max(trade_date)
            from ml_feature_mart_daily
            where feature_set_id = ? and trade_date between ? and ?
            """,
            [feature_set_id, start_date, end_date],
        ).fetchone()
        tradeability_count, tradeability_min, tradeability_max = con.execute(
            """
            select count(*), min(trade_date), max(trade_date)
            from ml_tradeability_daily
            where trade_date between ? and ?
            """,
            [start_date, end_date],
        ).fetchone()
        feature_duplicate_keys = con.execute(
            """
            select count(*)
            from (
                select trade_date, code, feature_set_id, count(*) as n
                from ml_feature_mart_daily
                where feature_set_id = ? and trade_date between ? and ?
                group by 1, 2, 3
                having n > 1
            )
            """,
            [feature_set_id, start_date, end_date],
        ).fetchone()[0]
        tradeability_duplicate_keys = con.execute(
            """
            select count(*)
            from (
                select trade_date, code, count(*) as n
                from ml_tradeability_daily
                where trade_date between ? and ?
                group by 1, 2
                having n > 1
            )
            """,
            [start_date, end_date],
        ).fetchone()[0]
    finally:
        con.close()

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ml_db": ml_db,
        "alpha_data_db": alpha_data_db,
        "vpa_db": vpa_db,
        "config_path": config_path,
        "start_date": start_date,
        "end_date": end_date,
        "feature_set_id": feature_set_id,
        "windows": windows,
        "warmup_months": warmup_months,
        "lookahead_days": lookahead_days,
        "batch_months": batch_months,
        "planned_batches": planned_batches,
        "feature_store_written": False,
        "tables": {
            "ml_feature_mart_daily": {
                "rows": int(feature_count),
                "min_date": feature_min,
                "max_date": feature_max,
                "duplicate_keys": int(feature_duplicate_keys),
            },
            "ml_tradeability_daily": {
                "rows": int(tradeability_count),
                "min_date": tradeability_min,
                "max_date": tradeability_max,
                "duplicate_keys": int(tradeability_duplicate_keys),
            },
        },
    }
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _keep_output_window(frame: pd.DataFrame, date_column: str, batch: BatchPeriod) -> pd.DataFrame:
    return frame[(frame[date_column] >= batch.out_start) & (frame[date_column] <= batch.out_end)].copy()


def _add_calendar_days(date: str, days: int) -> str:
    return (pd.Timestamp(date) + pd.DateOffset(days=days)).strftime("%Y-%m-%d")


def _safe_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"invalid SQL identifier: {value}")
    return value
