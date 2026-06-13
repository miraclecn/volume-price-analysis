from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
from dataclasses import dataclass, replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.backtest.engine import BacktestConfig, run_backtest, run_fixed_horizon_backtest, run_holding_aware_backtest
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
from ml_stock_selector.portfolio.fixed_horizon import fixed_horizon_config_from_dict
from ml_stock_selector.runtime.artifacts import write_backtest_fold_artifacts
from ml_stock_selector.runtime.run_context import create_run_context, register_run_context, register_run_fold, update_run_status
from ml_stock_selector.scoring import add_context_score, add_liquidity_score, score_candidates, score_candidates_v2
from ml_stock_selector.storage import clear_backtest_outputs, clear_portfolio_targets, init_ml_db, upsert_dataframe


SCORE_VERSION_THREE_MODEL = "v2_three_model"
SCORE_VERSION_ABSOLUTE_ONLY = "v2_absolute_only"
SCORE_VERSION_ABSOLUTE_RISK_FILTER = "v2_absolute_risk_filter"
SCORE_VERSION_ABSOLUTE_RISK_SORT = "v2_absolute_risk_sort"
STRATEGY_FIXED_5D_RISK_FILTER = "abs_ranker_fixed_5d_risk_filter_v1"
STRATEGY_FIXED_5D_NO_RISK_EXIT = "abs_ranker_fixed_5d_no_risk_exit_v1"
STRATEGY_HOLDING_AWARE_V2 = "holding_aware_v2"
STRATEGY_LEGACY_V1 = "legacy_target_rebalance_v1"


@dataclass(frozen=True)
class BacktestIdentity:
    run_id: str
    fold_id: str
    strategy_id: str
    score_version: str
    portfolio_id: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--run-id")
    parser.add_argument("--fold-id")
    parser.add_argument(
        "--score-mode",
        choices=["three_model", "absolute_only", "absolute_risk_filter", "absolute_risk_sort"],
        default="three_model",
    )
    parser.add_argument("--candidate-risk-max-rank-pct", type=float)
    parser.add_argument("--core-risk-max-rank-pct", type=float)
    parser.add_argument("--target-positions", type=int)
    parser.add_argument("--selection-buckets", type=int)
    parser.add_argument("--selection-per-bucket", type=int)
    parser.add_argument("--portfolio-suffix")
    parser.add_argument("--strategy-id")
    parser.add_argument("--ml-db")
    parser.add_argument("--score-version")
    parser.add_argument("--feature-set-id")
    parser.add_argument("--horizon-d", type=int)
    parser.add_argument("--label-base")
    parser.add_argument("--run-artifact-dir", default="outputs/ml/runs")
    return parser


def _score_version_for_mode(score_mode: str) -> str:
    if score_mode == "absolute_only":
        return SCORE_VERSION_ABSOLUTE_ONLY
    if score_mode == "absolute_risk_filter":
        return SCORE_VERSION_ABSOLUTE_RISK_FILTER
    if score_mode == "absolute_risk_sort":
        return SCORE_VERSION_ABSOLUTE_RISK_SORT
    return SCORE_VERSION_THREE_MODEL


def _portfolio_id_for_mode(
    fold_id: str | None,
    score_mode: str,
    suffix: str | None = None,
    strategy_id: str | None = None,
) -> str:
    base = fold_id or "default"
    if strategy_id:
        base = f"{base}_{strategy_id}"
    if score_mode == "absolute_only":
        base = f"{base}_absolute_only"
    elif score_mode == "absolute_risk_filter":
        base = f"{base}_absolute_risk_filter"
    elif score_mode == "absolute_risk_sort":
        base = f"{base}_absolute_risk_sort"
    if suffix:
        base = f"{base}_{suffix}"
    return base


def _backtest_identity(args) -> BacktestIdentity:
    run_id = args.run_id or "default_run"
    fold_id = args.fold_id or "default"
    explicit_strategy = args.strategy_id
    fixed_strategy = explicit_strategy in {STRATEGY_FIXED_5D_RISK_FILTER, STRATEGY_FIXED_5D_NO_RISK_EXIT}
    strategy_id = explicit_strategy or STRATEGY_HOLDING_AWARE_V2
    score_version = args.score_version or (strategy_id if fixed_strategy else _score_version_for_mode(args.score_mode))
    portfolio_id = _portfolio_id_for_mode(
        fold_id,
        "three_model" if fixed_strategy else args.score_mode,
        args.portfolio_suffix,
        strategy_id=strategy_id,
    )
    return BacktestIdentity(run_id, fold_id, strategy_id, score_version, portfolio_id)


def _apply_score_mode(scored: pd.DataFrame, score_mode: str) -> pd.DataFrame:
    out = scored.copy()
    if score_mode not in {"absolute_only", "absolute_risk_filter", "absolute_risk_sort"}:
        return out
    if "absolute_rank_pct" not in out:
        raise ValueError("absolute_only score mode requires absolute_rank_pct")
    absolute_rank = pd.to_numeric(out["absolute_rank_pct"], errors="coerce").fillna(0.0)
    risk_rank = pd.to_numeric(out["risk_rank_pct"], errors="coerce").fillna(1.0) if "risk_rank_pct" in out else 1.0
    trade_score = absolute_rank if score_mode != "absolute_risk_sort" else 0.85 * absolute_rank - 0.15 * risk_rank
    out["trade_score_v2"] = trade_score
    out["core_score"] = trade_score
    out["trade_score"] = trade_score
    out["alpha_rank_pct"] = absolute_rank
    out["active_rank_pct"] = absolute_rank
    if score_mode == "absolute_only":
        out["risk_rank_pct"] = 0.0
        if "risk_prob" in out:
            out["risk_prob"] = 0.0
    out["score_version"] = _score_version_for_mode(score_mode)
    return out


def _apply_constraint_overrides(constraints, args):
    updates = {}
    if args.score_mode == "absolute_risk_sort":
        updates["candidate_min_trade_score"] = 0.451
        updates["core_min_trade_score"] = 0.451
        updates["candidate_absolute_min_rank_pct"] = 0.75
        updates["core_absolute_min_rank_pct"] = 0.75
        updates["candidate_active_min_rank_pct"] = 0.75
        updates["core_active_min_rank_pct"] = 0.75
        updates["candidate_risk_max_rank_pct"] = 1.0
        updates["core_risk_max_rank_pct"] = 1.0
    if args.candidate_risk_max_rank_pct is not None:
        updates["candidate_risk_max_rank_pct"] = float(args.candidate_risk_max_rank_pct)
    if args.core_risk_max_rank_pct is not None:
        updates["core_risk_max_rank_pct"] = float(args.core_risk_max_rank_pct)
    if args.target_positions is not None:
        target_positions = int(args.target_positions)
        updates["target_positions"] = target_positions
        updates["hard_max_positions"] = max(int(constraints.hard_max_positions), target_positions)
        updates["max_initial_entries"] = max(int(constraints.max_initial_entries), target_positions)
    if args.selection_buckets is not None:
        updates["selection_bucket_count"] = int(args.selection_buckets)
    if args.selection_per_bucket is not None:
        updates["selection_per_bucket"] = int(args.selection_per_bucket)
    return replace(constraints, **updates) if updates else constraints


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_ml_config(args.config)
    ml_db = args.ml_db or str(config.data["ml_db"])
    con = init_ml_db(ml_db)
    try:
        preds = load_backtest_candidates(
            con,
            run_id=args.run_id,
            fold_id=args.fold_id,
            score_version=SCORE_VERSION_THREE_MODEL if bool(config.ml_v2["trade_score_v2_enabled"]) else None,
            exclude_bse=bool(config.universe.get("exclude_bse", False)),
        )
        if preds.empty:
            raise ValueError("No predictions matched the requested run/fold")
        identity = _backtest_identity(args)
        rid = identity.run_id
        score_version = identity.score_version
        portfolio_id = identity.portfolio_id
        fold_id = identity.fold_id
        strategy_id = identity.strategy_id
        context = create_run_context(
            run_type="backtest",
            run_id=rid,
            experiment_name=str(args.score_mode),
            config_path=args.config,
            artifact_root=args.run_artifact_dir,
            alpha_data_db=str(config.data["alpha_data_db"]),
            ml_db=ml_db,
            feature_set_id=args.feature_set_id or str(config.features["feature_set_id"]),
            label_version=f"{args.label_base or str(config.labels['label_base'])}_h{args.horizon_d or int(config.labels['main_horizon'])}",
            score_version=score_version,
        )
        register_run_context(con, context)
        fixed_strategy = strategy_id in {STRATEGY_FIXED_5D_RISK_FILTER, STRATEGY_FIXED_5D_NO_RISK_EXIT}
        strategy_params = {}
        if fixed_strategy:
            profile_name = "fixed_5d_no_risk_exit" if strategy_id == STRATEGY_FIXED_5D_NO_RISK_EXIT else "fixed_5d_risk_filter"
            loaded_constraints = fixed_horizon_config_from_dict(config.portfolio.get(profile_name, {}))
            fixed_constraints = replace(loaded_constraints, strategy_id=strategy_id)
            scored = preds.copy()
            if "absolute_rank_pct" not in scored and "alpha_rank_pct" in scored:
                scored["absolute_rank_pct"] = scored["alpha_rank_pct"]
            bars = load_normalized_stock_bars(str(config.data["alpha_data_db"]), scored["trade_date"].min(), scored["trade_date"].max(), str(config.data["normalized_bars_table"]))
            result = run_fixed_horizon_backtest(
                scored,
                bars,
                fixed_constraints,
                BacktestConfig(
                    float(config.backtest["initial_cash"]),
                    fixed_constraints.strategy_id,
                    ExecutionConfig(),
                    decision_dates=sorted(scored["trade_date"].dropna().unique()),
                ),
                run_id=rid,
                fold_id=fold_id,
            )
            targets = pd.DataFrame()
            diagnostics = result.portfolio_diagnostics if result.portfolio_diagnostics is not None else pd.DataFrame()
            strategy_params = fixed_constraints
        else:
            constraints = _apply_constraint_overrides(_portfolio_constraints_from_config(config), args)
            strategy_params = constraints
            if bool(config.ml_v2["trade_score_v2_enabled"]):
                scored = preds.copy()
                if "trade_score_v2" not in scored or scored["trade_score_v2"].isna().any():
                    scored = score_candidates_v2(add_liquidity_score(add_context_score(scored)))
                scored = _apply_score_mode(scored, args.score_mode)
                unweighted_targets = construct_portfolio_targets_v2(
                    scored,
                    constraints,
                    portfolio_id,
                    run_id=rid,
                    fold_id=fold_id,
                    score_version=score_version,
                )
                diagnostics = get_portfolio_diagnostics(unweighted_targets)
                targets = allocate_weights(
                    unweighted_targets,
                    float(config.portfolio["single_name_min_weight"]),
                    float(config.portfolio["single_name_max_weight"]),
                    bool(config.portfolio["allow_cash"]),
                )
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
                        fold_id,
                        ExecutionConfig(),
                        decision_dates=sorted(scored["trade_date"].dropna().unique()),
                    ),
                    min_weight=float(config.portfolio["single_name_min_weight"]),
                    max_weight=float(config.portfolio["single_name_max_weight"]),
                    allow_cash=bool(config.portfolio["allow_cash"]),
                    run_id=rid,
                    fold_id=fold_id,
                    score_version=score_version,
                )
            else:
                result = run_backtest(
                    targets,
                    bars,
                    BacktestConfig(
                        float(config.backtest["initial_cash"]),
                        fold_id,
                        ExecutionConfig(),
                        decision_dates=sorted(scored["trade_date"].dropna().unique()),
                    ),
                )
        targets = _annotate_targets(targets, rid, fold_id, portfolio_id, score_version)
        diagnostics = _annotate_diagnostics(diagnostics, rid, fold_id, portfolio_id, score_version)
        for frame in [result.nav, result.orders, result.positions]:
            if not frame.empty:
                frame["run_id"] = rid
                frame["fold_id"] = fold_id
                frame["strategy_id"] = strategy_id
                frame["score_version"] = score_version
        diagnostic_report_metrics = portfolio_diagnostics_report_metrics(diagnostics)
        metrics = summarize_fold_metric_rows(
            result,
            run_id=rid,
            fold_id=fold_id,
            score_version=score_version,
            strategy_id=strategy_id,
            start_date=str(scored["trade_date"].min()),
            end_date=str(scored["trade_date"].max()),
            candidate_pool_size=diagnostic_report_metrics["avg_candidate_pool_size"],
            core_pool_size=diagnostic_report_metrics["avg_core_pool_size"],
            bse_excluded_count=0.0,
        )
        start_date = str(scored["trade_date"].min())
        end_date = str(scored["trade_date"].max())
        register_run_fold(
            con,
            context,
            {
                "fold_id": fold_id,
                "test_start": start_date,
                "test_end": end_date,
            },
            status="success",
        )
        clear_backtest_outputs(con, rid, fold_id, strategy_id, score_version, start_date, end_date)
        clear_portfolio_targets(con, rid, fold_id, portfolio_id, score_version, start_date, end_date)
        if not targets.empty:
            upsert_dataframe(con, "ml_portfolio_targets_daily", targets, ["trade_date", "run_id", "fold_id", "portfolio_id", "score_version", "code"])
        if not diagnostics.empty and {"run_id", "fold_id", "portfolio_id", "score_version"}.issubset(diagnostics.columns):
            upsert_dataframe(con, "ml_portfolio_construction_diagnostics", diagnostics, ["trade_date", "run_id", "fold_id", "portfolio_id", "score_version"])
        upsert_dataframe(con, "ml_backtest_orders", result.orders, ["run_id", "fold_id", "strategy_id", "score_version", "sim_date", "decision_date", "code", "side", "order_seq"])
        upsert_dataframe(con, "ml_backtest_positions", result.positions, ["run_id", "fold_id", "strategy_id", "score_version", "sim_date", "code"])
        upsert_dataframe(con, "ml_backtest_nav", result.nav, ["run_id", "fold_id", "strategy_id", "score_version", "sim_date"])
        upsert_dataframe(con, "ml_backtest_metrics", metrics, ["run_id", "fold_id", "score_version", "metric_name", "segment"])
        if not diagnostics.empty:
            write_portfolio_diagnostics_report(
                diagnostics,
                str(config.data["report_dir"]),
                prefix=f"{portfolio_id}_portfolio_diagnostics",
            )
        write_backtest_fold_artifacts(
            context.artifact_root,
            fold_id=fold_id,
            strategy_id=strategy_id,
            score_version=score_version,
            portfolio_id=portfolio_id,
            backtest_params={
                "cli_args": vars(args),
                "strategy_params": strategy_params,
                "execution": ExecutionConfig(),
                "initial_cash": float(config.backtest["initial_cash"]),
                "start_date": start_date,
                "end_date": end_date,
            },
            targets=targets,
            diagnostics=diagnostics,
            orders=result.orders,
            positions=result.positions,
            nav=result.nav,
            metrics=metrics,
        )
        update_run_status(con, context, "success")
    finally:
        con.close()
    print(f"nav_rows={len(result.nav)}")


def _annotate_targets(
    targets: pd.DataFrame,
    run_id: str,
    fold_id: str,
    portfolio_id: str,
    score_version: str,
) -> pd.DataFrame:
    if targets.empty:
        return targets
    out = targets.copy()
    out["run_id"] = run_id
    out["fold_id"] = fold_id
    out["portfolio_id"] = portfolio_id
    out["score_version"] = score_version
    return out


def _annotate_diagnostics(
    diagnostics: pd.DataFrame,
    run_id: str,
    fold_id: str,
    portfolio_id: str,
    score_version: str,
) -> pd.DataFrame:
    if diagnostics.empty:
        return diagnostics
    out = diagnostics.copy()
    out["run_id"] = run_id
    out["fold_id"] = fold_id
    out["portfolio_id"] = portfolio_id
    out["score_version"] = score_version
    return out


if __name__ == "__main__":
    main()
