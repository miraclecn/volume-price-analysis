from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_stock_selector.config import load_ml_config
from ml_stock_selector.feature_mart_batch import (
    completed_feature_mart_month_keys,
    feature_mart_period_batches,
    run_feature_mart_batch,
    write_feature_mart_manifest,
)
from vpa_structure_recognizer.batch_runner import BatchPeriod, missing_month_batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML feature mart in resumable warmup batches")
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--alpha-data-db")
    parser.add_argument("--vpa-db")
    parser.add_argument("--ml-db", required=True)
    parser.add_argument("--normalized-bars-table")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--warmup-months", type=int, default=13)
    parser.add_argument("--batch-months", type=int, default=1)
    parser.add_argument("--lookahead-days", type=int, default=31)
    parser.add_argument("--manifest")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()

    config = load_ml_config(args.config)
    alpha_data_db = args.alpha_data_db or str(config.data["alpha_data_db"])
    vpa_db = args.vpa_db or str(config.data["vpa_db"])
    normalized_bars_table = args.normalized_bars_table or str(config.data["normalized_bars_table"])
    feature_set_id = str(config.features["feature_set_id"])
    windows = [int(value) for value in config.features["windows"]]
    exclude_industry_metadata = bool(config.ml_v2["exclude_industry_metadata_from_features_json"])

    if args.child:
        batch = BatchPeriod(
            (pd.Timestamp(args.start_date) - pd.DateOffset(months=args.warmup_months)).strftime("%Y-%m-%d"),
            pd.Timestamp(args.start_date).strftime("%Y-%m-%d"),
            pd.Timestamp(args.end_date).strftime("%Y-%m-%d"),
        )
        counts = run_feature_mart_batch(
            alpha_data_db=alpha_data_db,
            vpa_db=vpa_db,
            ml_db=args.ml_db,
            normalized_bars_table=normalized_bars_table,
            batch=batch,
            feature_set_id=feature_set_id,
            windows=windows,
            exclude_industry_metadata_from_features_json=exclude_industry_metadata,
            lookahead_days=args.lookahead_days,
        )
        print(json.dumps(counts, ensure_ascii=False, sort_keys=True), flush=True)
        return

    batches = feature_mart_period_batches(args.start_date, args.end_date, args.warmup_months, args.batch_months)
    completed = completed_feature_mart_month_keys(
        ml_db=args.ml_db,
        alpha_data_db=alpha_data_db,
        normalized_bars_table=normalized_bars_table,
        batches=batches,
        feature_set_id=feature_set_id,
    )
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
                "scripts/run_ml_feature_mart_batch.py",
                "--config",
                args.config,
                "--alpha-data-db",
                alpha_data_db,
                "--vpa-db",
                vpa_db,
                "--ml-db",
                args.ml_db,
                "--normalized-bars-table",
                normalized_bars_table,
                "--start-date",
                batch.out_start,
                "--end-date",
                batch.out_end,
                "--warmup-months",
                str(args.warmup_months),
                "--lookahead-days",
                str(args.lookahead_days),
                "--child",
            ],
            check=True,
            env=env,
        )

    manifest_path = args.manifest or str(Path(args.ml_db).with_suffix(".manifest.json"))
    manifest = write_feature_mart_manifest(
        manifest_path=manifest_path,
        ml_db=args.ml_db,
        alpha_data_db=alpha_data_db,
        vpa_db=vpa_db,
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        feature_set_id=feature_set_id,
        windows=windows,
        warmup_months=args.warmup_months,
        lookahead_days=args.lookahead_days,
        batch_months=args.batch_months,
        planned_batches=len(batches),
    )
    print(json.dumps({"manifest": manifest_path, "tables": manifest["tables"]}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
