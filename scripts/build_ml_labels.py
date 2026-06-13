from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.benchmarks import build_benchmark_tables
from ml_stock_selector.data_access import load_normalized_stock_bars
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()
    config = load_ml_config(args.config)
    start_date = args.start_date or "1900-01-01"
    end_date = args.end_date or "2999-12-31"
    horizons = [int(value) for value in config.labels["horizons"]]
    load_end_date = _label_load_end_date(end_date, horizons)
    bars = load_normalized_stock_bars(str(config.data["alpha_data_db"]), start_date, load_end_date, str(config.data["normalized_bars_table"]))
    labels = build_labels(
        bars,
        horizons,
        float(config.labels["risk_drawdown_threshold"]),
        include_v2=bool(config.ml_v2["labels_v2_enabled"]),
    )
    labels = _trim_labels_to_requested_range(labels, start_date, end_date)
    con = init_ml_db(str(config.data["ml_db"]))
    try:
        upsert_dataframe(con, "ml_labels_daily", labels, ["trade_date", "code", "horizon_d", "label_base"])
        market_bm, industry_bm = build_benchmark_tables(labels)
        upsert_dataframe(con, "ml_market_benchmark_daily", market_bm, ["trade_date", "horizon_d", "label_base"])
        upsert_dataframe(con, "ml_industry_benchmark_daily", industry_bm, ["trade_date", "industry_code", "horizon_d", "label_base"])
    finally:
        con.close()
    print(f"rows={len(labels)}")


def _label_load_end_date(end_date: str, horizons: list[int]) -> str:
    parsed = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(parsed):
        return end_date
    max_horizon = max([int(value) for value in horizons], default=0)
    if max_horizon <= 0:
        return end_date
    lookahead_days = max(30, max_horizon * 4 + 14)
    try:
        return (parsed + timedelta(days=lookahead_days)).strftime("%Y-%m-%d")
    except (OverflowError, ValueError):
        return end_date


def _trim_labels_to_requested_range(labels, start_date: str, end_date: str):
    if labels.empty or "trade_date" not in labels:
        return labels
    mask = (labels["trade_date"].astype(str) >= str(start_date)) & (labels["trade_date"].astype(str) <= str(end_date))
    return labels.loc[mask].copy()


if __name__ == "__main__":
    main()
