from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
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
    constraints = PortfolioConstraints(
        min_trade_score=float(config.portfolio["min_trade_score"]),
        min_adv20_amount=float(config.portfolio.get("min_adv20_amount", 0.0)) or None,
        target_positions=int(config.portfolio["target_positions"]),
        hard_max_positions=int(config.portfolio["hard_max_positions"]),
        max_industry_names=int(config.portfolio["max_industry_names"]),
        max_new_entries_per_day=int(config.portfolio["max_new_entries_per_day"]),
        allow_cash=bool(config.portfolio["allow_cash"]),
        candidate_absolute_min_rank_pct=float(config.ml_v2["candidate_absolute_min_rank_pct"]),
        candidate_active_min_rank_pct=float(config.ml_v2["candidate_active_min_rank_pct"]),
        candidate_risk_max_rank_pct=float(config.ml_v2["candidate_risk_max_rank_pct"]),
        core_absolute_min_rank_pct=float(config.ml_v2["core_absolute_min_rank_pct"]),
        core_active_min_rank_pct=float(config.ml_v2["core_active_min_rank_pct"]),
        core_risk_max_rank_pct=float(config.ml_v2["core_risk_max_rank_pct"]),
        core_min_trade_score=float(config.ml_v2["core_min_trade_score"]),
    )
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
        )
    finally:
        con.close()
    print(f"predictions={len(predictions)} targets={len(targets)}")


if __name__ == "__main__":
    main()
