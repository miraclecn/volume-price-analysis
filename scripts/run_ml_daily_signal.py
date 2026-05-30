from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.serving.daily_signal import generate_daily_signal
from ml_stock_selector.storage import init_ml_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--as-of-date")
    parser.add_argument("--portfolio-id", default="default")
    args = parser.parse_args()
    config = load_ml_config(args.config)
    con = init_ml_db(str(config.data["ml_db"]))
    try:
        predictions, targets = generate_daily_signal(
            con,
            args.as_of_date,
            str(config.features["feature_set_id"]),
            int(config.labels["main_horizon"]),
            args.portfolio_id,
        )
    finally:
        con.close()
    print(f"predictions={len(predictions)} targets={len(targets)}")


if __name__ == "__main__":
    main()
