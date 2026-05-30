from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.backtest.walkforward import run_walkforward_experiment
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.data_access import load_normalized_stock_bars
from ml_stock_selector.storage import init_ml_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_default.toml")
    args = parser.parse_args()
    config = load_ml_config(args.config)
    con = init_ml_db(str(config.data["ml_db"]))
    try:
        feature_mart = con.execute("select * from ml_feature_mart_daily").fetchdf()
        labels = con.execute("select * from ml_labels_daily").fetchdf()
        tradeability = con.execute("select * from ml_tradeability_daily").fetchdf()
        bars = load_normalized_stock_bars(str(config.data["alpha_data_db"]), feature_mart["trade_date"].min(), "2999-12-31", str(config.data["normalized_bars_table"]))
        results = run_walkforward_experiment(config, bars, feature_mart, labels, tradeability, str(config.data["artifact_dir"]))
    finally:
        con.close()
    print(f"folds={len(results)}")


if __name__ == "__main__":
    main()
