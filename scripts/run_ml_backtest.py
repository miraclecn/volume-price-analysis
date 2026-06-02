from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.backtest.engine import BacktestConfig, run_backtest, run_holding_aware_backtest
from ml_stock_selector.backtest.execution import ExecutionConfig
from ml_stock_selector.backtest.data_access import load_backtest_candidates
from ml_stock_selector.backtest.metrics import summarize_fold_metric_rows
from ml_stock_selector.backtest.reports import (
    portfolio_diagnostics_report_metrics,
    write_portfolio_diagnostics_report,
)
from ml_stock_selector.backtest.walkforward import _portfolio_constraints_from_config
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.data_access import load_normalized_stock_bars
from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constructor import (
    construct_portfolio_targets,
    construct_portfolio_targets_v2,
    get_portfolio_diagnostics,
)
from ml_stock_selector.scoring import add_context_score, add_liquidity_score, score_candidates, score_candidates_v2
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--run-id")
    parser.add_argument("--fold-id")
    args = parser.parse_args()
    config = load_ml_config(args.config)
    con = init_ml_db(str(config.data["ml_db"]))
    try:
        preds = load_backtest_candidates(
            con,
            run_id=args.run_id,
            fold_id=args.fold_id,
            score_version="v2_three_model" if bool(config.ml_v2["trade_score_v2_enabled"]) else None,
            exclude_bse=bool(config.universe.get("exclude_bse", False)),
        )
        if preds.empty:
            raise ValueError("No predictions matched the requested run/fold")
        constraints = _portfolio_constraints_from_config(config)
        rid = args.run_id or "default_run"
        portfolio_id = args.fold_id or "default"
        if bool(config.ml_v2["trade_score_v2_enabled"]):
            scored = preds.copy()
            if "trade_score_v2" not in scored or scored["trade_score_v2"].isna().any():
                scored = score_candidates_v2(add_liquidity_score(add_context_score(scored)))
            unweighted_targets = construct_portfolio_targets_v2(
                scored,
                constraints,
                portfolio_id,
                run_id=rid,
                fold_id=portfolio_id,
                score_version="v2_three_model",
            )
            diagnostics = get_portfolio_diagnostics(unweighted_targets)
            targets = allocate_weights(unweighted_targets, 0.05, 0.10, bool(config.portfolio["allow_cash"]))
        else:
            scored = preds.copy()
            if "trade_score" not in scored or scored["trade_score"].isna().any():
                scored = score_candidates(add_liquidity_score(add_context_score(scored)))
            targets = allocate_weights(construct_portfolio_targets(scored, constraints, portfolio_id), 0.05, 0.10, bool(config.portfolio["allow_cash"]))
            diagnostics = pd.DataFrame()
        bars = load_normalized_stock_bars(str(config.data["alpha_data_db"]), scored["trade_date"].min(), scored["trade_date"].max(), str(config.data["normalized_bars_table"]))
        if bool(config.ml_v2["trade_score_v2_enabled"]):
            result = run_holding_aware_backtest(
                scored,
                bars,
                constraints,
                BacktestConfig(
                    float(config.backtest["initial_cash"]),
                    args.fold_id or "default",
                    ExecutionConfig(),
                    decision_dates=sorted(scored["trade_date"].dropna().unique()),
                ),
                min_weight=0.05,
                max_weight=0.10,
                allow_cash=bool(config.portfolio["allow_cash"]),
                run_id=rid,
                fold_id=portfolio_id,
                score_version="v2_three_model",
            )
        else:
            result = run_backtest(
                targets,
                bars,
                BacktestConfig(
                    float(config.backtest["initial_cash"]),
                    args.fold_id or "default",
                    ExecutionConfig(),
                    decision_dates=sorted(scored["trade_date"].dropna().unique()),
                ),
            )
        result.nav["run_id"] = rid
        result.orders["run_id"] = rid
        result.positions["run_id"] = rid
        diagnostic_report_metrics = portfolio_diagnostics_report_metrics(diagnostics)
        metrics = summarize_fold_metric_rows(
            result,
            run_id=rid,
            fold_id=args.fold_id or "default",
            score_version="v2_three_model",
            strategy_id=args.fold_id or "default",
            start_date=str(scored["trade_date"].min()),
            end_date=str(scored["trade_date"].max()),
            candidate_pool_size=diagnostic_report_metrics["avg_candidate_pool_size"],
            core_pool_size=diagnostic_report_metrics["avg_core_pool_size"],
            bse_excluded_count=0.0,
        )
        upsert_dataframe(con, "ml_portfolio_targets_daily", targets, ["trade_date", "portfolio_id", "code"])
        upsert_dataframe(con, "ml_portfolio_construction_diagnostics", diagnostics, ["trade_date", "run_id", "fold_id", "portfolio_id", "score_version"])
        upsert_dataframe(con, "ml_backtest_orders", result.orders, ["sim_date", "decision_date", "code", "side"])
        upsert_dataframe(con, "ml_backtest_positions", result.positions, ["sim_date", "code"])
        upsert_dataframe(con, "ml_backtest_nav", result.nav, ["sim_date"])
        upsert_dataframe(con, "ml_backtest_metrics", metrics, ["run_id", "fold_id", "score_version", "metric_name", "segment"])
        if not diagnostics.empty:
            write_portfolio_diagnostics_report(
                diagnostics,
                str(config.data["report_dir"]),
                prefix=f"{portfolio_id}_portfolio_diagnostics",
            )
    finally:
        con.close()
    print(f"nav_rows={len(result.nav)}")


if __name__ == "__main__":
    main()
