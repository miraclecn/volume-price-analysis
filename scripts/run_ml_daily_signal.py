from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.feature_store_reader import FeatureStoreSpec
from ml_stock_selector.backtest.walkforward import _portfolio_constraints_from_config
from ml_stock_selector.serving.daily_signal import generate_daily_signal
from ml_stock_selector.storage import init_ml_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--as-of-date")
    parser.add_argument("--portfolio-id", default="default")
    parser.add_argument("--feature-store-dir")
    parser.add_argument("--feature-store-version")
    parser.add_argument("--use-feature-store", type=_parse_bool, default=True)
    args = parser.parse_args()
    config = load_ml_config(args.config)
    con = init_ml_db(str(config.data["ml_db"]))
    constraints = _portfolio_constraints_from_config(config)
    try:
        predictions, targets = generate_daily_signal(
            con,
            args.as_of_date,
            str(config.features["feature_set_id"]),
            int(config.labels["main_horizon"]),
            args.portfolio_id,
            constraints=constraints,
            use_v2=bool(config.ml_v2["daily_signal_v2_enabled"]),
            exclude_bse=bool(config.universe.get("exclude_bse", False)),
            feature_store_spec=FeatureStoreSpec(
                args.feature_store_dir or "outputs/ml/feature_store",
                args.feature_store_version or "v2_pv_only_001",
                str(config.features["feature_set_id"]),
            )
            if args.use_feature_store
            else None,
        )
    finally:
        con.close()
    print(f"predictions={len(predictions)} targets={len(targets)}")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


if __name__ == "__main__":
    main()
