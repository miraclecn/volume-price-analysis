from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.backtest.engine import BacktestConfig, run_backtest
from ml_stock_selector.backtest.execution import ExecutionConfig
from ml_stock_selector.backtest.metrics import annualized_return, max_drawdown
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.data_access import load_normalized_stock_bars
from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.constructor import construct_portfolio_targets, construct_portfolio_targets_v2
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
        where = []
        params: list[object] = []
        if args.run_id:
            where.append("run_id = ?")
            params.append(args.run_id)
        if args.fold_id:
            where.append("fold_id = ?")
            params.append(args.fold_id)
        sql = "select * from ml_predictions_daily"
        if where:
            sql += " where " + " and ".join(where)
        preds = con.execute(sql, params).fetchdf()
        if not preds.empty:
            required_meta = {"is_st", "is_paused", "adv20_amount", "can_buy_next_open", "can_sell_next_open", "industry_code", "industry_name", "is_bse"}
            if not required_meta.issubset(set(preds.columns)) or preds[list(required_meta & set(preds.columns))].isna().any().any():
                keys = preds[["trade_date", "code"]].drop_duplicates()
                con.register("_pred_keys", keys)
                tradeability = con.execute(
                    """
                    select t.trade_date, t.code, t.industry_code, t.industry_name, t.is_st, t.is_paused,
                           t.adv20_amount, t.can_buy_next_open, t.can_sell_next_open, t.is_bse
                    from ml_tradeability_daily t
                    join _pred_keys k on t.trade_date = k.trade_date and t.code = k.code
                    """
                ).fetchdf()
                con.unregister("_pred_keys")
                preds = preds.drop(columns=[c for c in tradeability.columns if c in preds.columns and c not in {"trade_date", "code"}], errors="ignore")
                preds = preds.merge(tradeability, on=["trade_date", "code"], how="left")
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
        if bool(config.ml_v2["trade_score_v2_enabled"]):
            scored = score_candidates_v2(add_liquidity_score(add_context_score(preds)))
            targets = allocate_weights(construct_portfolio_targets_v2(scored, constraints, args.fold_id or "default"), 0.05, 0.10, bool(config.portfolio["allow_cash"]))
        else:
            scored = score_candidates(add_liquidity_score(add_context_score(preds)))
            targets = allocate_weights(construct_portfolio_targets(scored, constraints, args.fold_id or "default"), 0.05, 0.10, bool(config.portfolio["allow_cash"]))
        bars = load_normalized_stock_bars(str(config.data["alpha_data_db"]), scored["trade_date"].min(), "2999-12-31", str(config.data["normalized_bars_table"]))
        result = run_backtest(targets, bars, BacktestConfig(float(config.backtest["initial_cash"]), args.fold_id or "default", ExecutionConfig()))
        rid = args.run_id or "default_run"
        result.nav["run_id"] = rid
        result.orders["run_id"] = rid
        result.positions["run_id"] = rid
        metrics = pd.DataFrame(
            [
                {"run_id": rid, "fold_id": args.fold_id, "score_version": "v2_three_model", "metric_name": "annualized_return", "metric_value": annualized_return(result.nav), "segment": "run"},
                {"run_id": rid, "fold_id": args.fold_id, "score_version": "v2_three_model", "metric_name": "max_drawdown", "metric_value": max_drawdown(result.nav), "segment": "run"},
            ]
        )
        upsert_dataframe(con, "ml_portfolio_targets_daily", targets, ["trade_date", "portfolio_id", "code"])
        upsert_dataframe(con, "ml_backtest_orders", result.orders, ["sim_date", "decision_date", "code", "side"])
        upsert_dataframe(con, "ml_backtest_positions", result.positions, ["sim_date", "code"])
        upsert_dataframe(con, "ml_backtest_nav", result.nav, ["sim_date"])
        upsert_dataframe(con, "ml_backtest_metrics", metrics, ["run_id", "fold_id", "score_version", "metric_name", "segment"])
    finally:
        con.close()
    print(f"nav_rows={len(result.nav)}")


if __name__ == "__main__":
    main()
