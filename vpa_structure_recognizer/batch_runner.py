from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import duckdb
import pandas as pd

from vpa_structure_recognizer.bar_labeler import label_bars
from vpa_structure_recognizer.config import load_config
from vpa_structure_recognizer.data_sources import ResearchSourceDuckDB
from vpa_structure_recognizer.feature_engineering import compute_features
from vpa_structure_recognizer.market_aggregates import build_market_bars, build_sector_bars
from vpa_structure_recognizer.pipeline import _market_feature_input, _stock_sector_map
from vpa_structure_recognizer.sequence_analyzer import analyze_sequences
from vpa_structure_recognizer.state_classifier import classify_structure_states
from vpa_structure_recognizer.storage import init_vpa_db, upsert_dataframe
from vpa_structure_recognizer.top_down_ranker import rank_top_down
from vpa_structure_recognizer.trend_context import compute_trend_context


@dataclass(frozen=True)
class BatchPeriod:
    pre_start: str
    out_start: str
    out_end: str

    @property
    def month_key(self) -> str:
        return self.out_start[:7]


def month_batches(start_date: str, end_date: str, warmup_months: int = 13) -> list[BatchPeriod]:
    start = pd.Timestamp(start_date).replace(day=1)
    end = pd.Timestamp(end_date)
    batches: list[BatchPeriod] = []
    month = start
    while month <= end:
        month_end = min(month + pd.offsets.MonthEnd(0), end)
        pre_start = month - pd.DateOffset(months=warmup_months)
        batches.append(
            BatchPeriod(
                pre_start.strftime("%Y-%m-%d"),
                month.strftime("%Y-%m-%d"),
                month_end.strftime("%Y-%m-%d"),
            )
        )
        month = month + pd.DateOffset(months=1)
    return batches


def missing_month_batches(
    batches: list[BatchPeriod],
    completed_month_keys: set[str],
) -> list[BatchPeriod]:
    return [batch for batch in batches if batch.month_key not in completed_month_keys]


def completed_month_keys(
    *,
    output_db: str,
    source_db: str,
    batches: list[BatchPeriod],
) -> set[str]:
    if not Path(output_db).exists():
        return set()

    out = duckdb.connect(output_db, read_only=True)
    src = duckdb.connect(source_db, read_only=True)
    completed: set[str] = set()
    try:
        tables = {
            row[0]
            for row in out.execute(
                "select table_name from information_schema.tables where table_schema = 'main'"
            ).fetchall()
        }
        if "vpa_structure_state" not in tables:
            return set()

        start = min(batch.out_start for batch in batches)
        end = max(batch.out_end for batch in batches)
        source_dates = {
            row[0]
            for row in src.execute(
                """
                select distinct
                    case
                        when length(cast(trade_date as varchar)) = 8
                            then substr(cast(trade_date as varchar), 1, 4)
                                || '-' || substr(cast(trade_date as varchar), 5, 2)
                                || '-' || substr(cast(trade_date as varchar), 7, 2)
                        else cast(trade_date as varchar)
                    end as date
                from stock_bar_normalized_daily
                where trade_date between ? and ?
                """,
                [start.replace("-", ""), end.replace("-", "")],
            ).fetchall()
        }
        state_dates = {
            row[0]
            for row in out.execute(
                """
                select distinct date
                from vpa_structure_state
                where date between ? and ?
                """,
                [start, end],
            ).fetchall()
        }
        for batch in batches:
            expected = {
                date for date in source_dates if batch.out_start <= date <= batch.out_end
            }
            actual = {date for date in state_dates if batch.out_start <= date <= batch.out_end}
            if expected and expected <= actual:
                completed.add(batch.month_key)
    finally:
        out.close()
        src.close()
    return completed


def run_batch(
    *,
    config_path: str,
    source_db: str,
    output_db: str,
    batch: BatchPeriod,
) -> dict[str, int]:
    config = load_config(config_path)
    source_reader = ResearchSourceDuckDB(source_db)

    stock_bars = source_reader.fetch_stock_bars(batch.pre_start, batch.out_end)
    source_row_count = len(stock_bars)
    sector_bars = build_sector_bars(stock_bars)
    market_bars = build_market_bars(stock_bars)
    features_all = pd.concat(
        [
            compute_features(stock_bars, config.windows, "stock", scope_id_column="code"),
            compute_features(sector_bars, config.windows, "sector", scope_id_column="sector_code"),
            compute_features(
                _market_feature_input(market_bars),
                config.windows,
                "market",
                scope_id="ALL_A",
            ),
        ],
        ignore_index=True,
    )
    features = _keep_output_window(features_all, batch)
    trend = compute_trend_context(features, config.parent_windows)
    labels_all = label_bars(features_all, config.parent_windows)
    labels = _keep_output_window(labels_all, batch)
    del features_all
    gc.collect()

    sequences_all = analyze_sequences(labels_all, trend)
    sequences = _keep_output_window(sequences_all, batch)
    del labels_all
    gc.collect()

    states = classify_structure_states(sequences, trend)
    del sequences_all
    gc.collect()

    ranked = rank_top_down(
        states,
        _stock_sector_map(stock_bars),
        config.scoring_weights,
    )
    del states, stock_bars, sector_bars, market_bars
    gc.collect()

    con = init_vpa_db(output_db)
    try:
        con.execute("pragma threads=1")
        upsert_dataframe(con, "vpa_features", features, ["date", "scope_type", "scope_id", "window_n"])
        upsert_dataframe(
            con,
            "vpa_trend_context",
            trend,
            ["date", "scope_type", "scope_id", "window_n"],
        )
        upsert_dataframe(
            con,
            "vpa_bar_context_labels",
            labels,
            ["date", "scope_type", "scope_id", "window_n"],
        )
        upsert_dataframe(
            con,
            "vpa_sequence_stats",
            sequences,
            ["date", "scope_type", "scope_id", "window_n"],
        )
        upsert_dataframe(con, "vpa_structure_state", ranked, ["date", "scope_type", "scope_id"])
        con.commit()
    finally:
        con.close()

    return {
        "source_rows": source_row_count,
        "features": len(features),
        "states": len(ranked),
    }


def _keep_output_window(frame: pd.DataFrame, batch: BatchPeriod) -> pd.DataFrame:
    return frame[(frame["date"] >= batch.out_start) & (frame["date"] <= batch.out_end)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VPA generation in resumable monthly child processes")
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--source", default="/home/nan/alpha-data-local/output/research_source.duckdb")
    parser.add_argument("--output-db", default="outputs/vpa.duckdb")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--warmup-months", type=int, default=13)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()

    if args.child:
        batch = BatchPeriod(
            (pd.Timestamp(args.start_date) - pd.DateOffset(months=args.warmup_months)).strftime("%Y-%m-%d"),
            pd.Timestamp(args.start_date).strftime("%Y-%m-%d"),
            pd.Timestamp(args.end_date).strftime("%Y-%m-%d"),
        )
        counts = run_batch(
            config_path=args.config,
            source_db=args.source,
            output_db=args.output_db,
            batch=batch,
        )
        print(f"DONE {batch.out_start}..{batch.out_end} {counts}", flush=True)
        return

    batches = month_batches(args.start_date, args.end_date, args.warmup_months)
    completed = completed_month_keys(output_db=args.output_db, source_db=args.source, batches=batches)
    remaining = missing_month_batches(batches, completed)
    print(f"planned={len(batches)} completed={len(completed)} remaining={len(remaining)}", flush=True)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    for index, batch in enumerate(remaining, 1):
        print(
            f"BATCH {index}/{len(remaining)} pre={batch.pre_start} "
            f"out={batch.out_start}..{batch.out_end}",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "vpa_structure_recognizer.batch_runner",
                "--config",
                args.config,
                "--source",
                args.source,
                "--output-db",
                args.output_db,
                "--start-date",
                batch.out_start,
                "--end-date",
                batch.out_end,
                "--warmup-months",
                str(args.warmup_months),
                "--child",
            ],
            check=True,
            env=env,
        )


if __name__ == "__main__":
    main()
