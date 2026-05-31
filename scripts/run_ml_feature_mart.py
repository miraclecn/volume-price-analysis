from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.data_access import load_normalized_stock_bars
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.storage import init_ml_db, upsert_dataframe
from ml_stock_selector.tradeability import build_tradeability_mart


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()
    config = load_ml_config(args.config)
    start_date = args.start_date or "1900-01-01"
    end_date = args.end_date or "2999-12-31"
    bars = load_normalized_stock_bars(str(config.data["alpha_data_db"]), start_date, end_date, str(config.data["normalized_bars_table"]))
    tradeability = build_tradeability_mart(bars)
    feature_mart = build_feature_mart(
        str(config.data["vpa_db"]),
        bars,
        start_date,
        end_date,
        str(config.features["feature_set_id"]),
        [int(value) for value in config.features["windows"]],
        tradeability,
        exclude_industry_metadata_from_features_json=bool(config.ml_v2["exclude_industry_metadata_from_features_json"]),
    )
    con = init_ml_db(str(config.data["ml_db"]))
    try:
        upsert_dataframe(con, "ml_tradeability_daily", tradeability, ["trade_date", "code"])
        upsert_dataframe(con, "ml_feature_mart_daily", feature_mart, ["trade_date", "code", "feature_set_id"])
    finally:
        con.close()
    print(f"rows={len(feature_mart)}")


if __name__ == "__main__":
    main()
