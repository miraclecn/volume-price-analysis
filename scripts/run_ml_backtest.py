from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.backtest.engine import BacktestConfig, run_backtest
from ml_stock_selector.backtest.execution import ExecutionConfig
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.data_access import load_normalized_stock_bars
from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.constructor import construct_portfolio_targets
from ml_stock_selector.scoring import add_context_score, add_liquidity_score, score_candidates
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = load_ml_config(args.config)
    con = init_ml_db(str(config.data["ml_db"]))
    try:
        preds = con.execute("select * from ml_predictions_daily").fetchdf()
        scored = score_candidates(add_liquidity_score(add_context_score(preds)))
        targets = allocate_weights(construct_portfolio_targets(scored, PortfolioConstraints(min_trade_score=-999.0), "default"), 0.05, 0.10, True)
        bars = load_normalized_stock_bars(str(config.data["alpha_data_db"]), scored["trade_date"].min(), "2999-12-31", str(config.data["normalized_bars_table"]))
        result = run_backtest(targets, bars, BacktestConfig(float(config.backtest["initial_cash"]), "default", ExecutionConfig()))
        upsert_dataframe(con, "ml_portfolio_targets_daily", targets, ["trade_date", "portfolio_id", "code"])
        upsert_dataframe(con, "ml_backtest_orders", result.orders, ["sim_date", "decision_date", "code", "side"])
        upsert_dataframe(con, "ml_backtest_positions", result.positions, ["sim_date", "code"])
        upsert_dataframe(con, "ml_backtest_nav", result.nav, ["sim_date"])
    finally:
        con.close()
    print(f"nav_rows={len(result.nav)}")


if __name__ == "__main__":
    main()
