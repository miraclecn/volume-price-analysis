from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.backtest.walkforward import run_walkforward_experiment, run_walkforward_feature_store_experiment
from ml_stock_selector.backtest.metrics import annualized_return, max_drawdown
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.data_access import load_normalized_stock_bars
from ml_stock_selector.portfolio.constructor import get_portfolio_diagnostics
from ml_stock_selector.runtime.run_context import create_run_context, register_run_context, register_run_fold, update_run_status
from ml_stock_selector.split.fold_generator import generate_experiment_folds
from ml_stock_selector.storage import init_ml_db, upsert_dataframe

STAGES = ["matrix", "models", "predictions", "portfolio", "backtest"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--experiment-name")
    parser.add_argument("--ml-db")
    parser.add_argument("--run-id")
    parser.add_argument("--fold-id")
    parser.add_argument("--feature-store-dir")
    parser.add_argument("--feature-store-version")
    parser.add_argument("--use-feature-store", type=_parse_bool, default=False)
    parser.add_argument("--matrix-cache-dir", default="outputs/ml/cache/folds")
    parser.add_argument("--feature-set-id")
    parser.add_argument("--horizon-d", type=int)
    parser.add_argument("--label-base")
    parser.add_argument("--score-version", default="v2_three_model")
    parser.add_argument("--run-artifact-dir", default="outputs/ml/runs")
    parser.add_argument("--from-stage", choices=STAGES, default="matrix")
    parser.add_argument("--to-stage", choices=STAGES, default="backtest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--generated-train-start", default="2015-01-05")
    parser.add_argument("--generated-first-test-year", type=int)
    parser.add_argument("--generated-last-test-year", type=int)
    parser.add_argument("--generated-last-test-end")
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--prediction-chunk-size", type=int, default=50000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-legacy-json-path", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_stage_range(args.from_stage, args.to_stage)
    config = load_ml_config(args.config)
    ml_db = args.ml_db or str(config.data["ml_db"])
    selected = resolve_selected_folds(args, config.split.get("folds", []))
    if not selected:
        raise ValueError(f"Unknown fold_id: {args.fold_id}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": args.config,
                    "run_id": args.run_id,
                    "experiment_name": args.experiment_name,
                    "from_stage": args.from_stage,
                    "to_stage": args.to_stage,
                    "folds": selected,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return
    con = init_ml_db(ml_db)
    try:
        if args.use_feature_store:
            run_id = args.run_id or "wf_three_model_v2_parquet_001"
            feature_set_id = args.feature_set_id or str(config.features["feature_set_id"])
            horizon_d = args.horizon_d or int(config.labels["main_horizon"])
            label_base = args.label_base or str(config.labels["label_base"])
            if not args.feature_store_dir or not args.feature_store_version:
                raise ValueError("--feature-store-dir and --feature-store-version are required when --use-feature-store true")
            if not selected:
                raise ValueError(f"Unknown fold_id: {args.fold_id}")
            context = create_run_context(
                run_type="walkforward",
                run_id=run_id,
                experiment_name=args.experiment_name or "feature_store_walkforward",
                config_path=args.config,
                artifact_root=args.run_artifact_dir,
                alpha_data_db=str(config.data["alpha_data_db"]),
                ml_db=ml_db,
                feature_set_id=feature_set_id,
                feature_store_version=args.feature_store_version,
                label_version=f"{label_base}_h{horizon_d}",
                score_version=args.score_version,
            )
            register_run_context(con, context)
            for fold in selected:
                register_run_fold(con, context, fold, status="running")
            bars = load_normalized_stock_bars(
                str(config.data["alpha_data_db"]),
                min(str(fold["test_start"]) for fold in selected),
                max(str(fold["test_end"]) for fold in selected),
                str(config.data["normalized_bars_table"]),
            )
            results = run_walkforward_feature_store_experiment(
                config,
                con,
                bars,
                run_id=run_id,
                feature_store_dir=args.feature_store_dir,
                feature_store_version=args.feature_store_version,
                matrix_cache_dir=args.matrix_cache_dir,
                feature_set_id=feature_set_id,
                horizon_d=horizon_d,
                label_base=label_base,
                score_version=args.score_version,
                fold_id=args.fold_id,
                artifact_dir=str(config.data["artifact_dir"]),
                run_artifact_root=context.artifact_root,
                batch_size=args.batch_size,
                prediction_chunk_size=args.prediction_chunk_size,
                force=args.force,
            )
        else:
            if not args.allow_legacy_json_path:
                raise ValueError("production-scale walk-forward must use Parquet Feature Store; pass --allow-legacy-json-path for small legacy JSON runs")
            print("features_json training path is legacy and not recommended for production-scale walk-forward")
            run_id = args.run_id or "legacy_walkforward"
            feature_set_id = args.feature_set_id or str(config.features["feature_set_id"])
            horizon_d = args.horizon_d or int(config.labels["main_horizon"])
            label_base = args.label_base or str(config.labels["label_base"])
            context = create_run_context(
                run_type="walkforward",
                run_id=run_id,
                experiment_name=args.experiment_name or "legacy_json_walkforward",
                config_path=args.config,
                artifact_root=args.run_artifact_dir,
                alpha_data_db=str(config.data["alpha_data_db"]),
                ml_db=ml_db,
                feature_set_id=feature_set_id,
                label_version=f"{label_base}_h{horizon_d}",
                score_version=args.score_version,
            )
            register_run_context(con, context)
            for fold in selected:
                register_run_fold(con, context, fold, status="running")
            feature_mart = con.execute("select * from ml_feature_mart_daily").fetchdf()
            labels = con.execute("select * from ml_labels_daily").fetchdf()
            tradeability = con.execute("select * from ml_tradeability_daily").fetchdf()
            bars = load_normalized_stock_bars(str(config.data["alpha_data_db"]), feature_mart["trade_date"].min(), "2999-12-31", str(config.data["normalized_bars_table"]))
            results = run_walkforward_experiment(
                config,
                con,
                bars,
                feature_mart,
                labels,
                tradeability,
                str(config.data["artifact_dir"]),
                run_id=run_id,
                run_artifact_root=context.artifact_root,
            )
        for item in results:
            result_fold = next((fold for fold in config.split.get("folds", []) if str(fold.get("fold_id")) == item.fold_id), {"fold_id": item.fold_id})
            register_run_fold(con, context, result_fold, status="success")
            if not args.use_feature_store:
                upsert_dataframe(con, "ml_predictions_daily", item.predictions, ["trade_date", "code", "model_id", "horizon_d"])
            target_score_version = str(item.metrics.get("score_version") or args.score_version)
            targets = _annotate_targets(
                item.targets,
                str(item.metrics.get("run_id") or args.run_id or "legacy_walkforward"),
                item.fold_id,
                target_score_version,
            )
            upsert_dataframe(con, "ml_portfolio_targets_daily", targets, ["trade_date", "run_id", "fold_id", "portfolio_id", "score_version", "code"])
            diagnostics = item.backtest_result.portfolio_diagnostics
            if diagnostics is None or diagnostics.empty:
                diagnostics = get_portfolio_diagnostics(targets)
            upsert_dataframe(con, "ml_portfolio_construction_diagnostics", diagnostics, ["trade_date", "run_id", "fold_id", "portfolio_id", "score_version"])
            metrics = [
                {"run_id": item.metrics.get("run_id"), "fold_id": item.fold_id, "score_version": "v2_three_model", "metric_name": "annualized_return", "metric_value": annualized_return(item.backtest_result.nav), "segment": "fold"},
                {"run_id": item.metrics.get("run_id"), "fold_id": item.fold_id, "score_version": "v2_three_model", "metric_name": "max_drawdown", "metric_value": max_drawdown(item.backtest_result.nav), "segment": "fold"},
            ]
            upsert_dataframe(con, "ml_backtest_metrics", pd.DataFrame(metrics), ["run_id", "fold_id", "score_version", "metric_name", "segment"])
        update_run_status(con, context, "success")
    finally:
        con.close()
    print(f"folds={len(results)}")


def validate_stage_range(from_stage: str, to_stage: str) -> None:
    if STAGES.index(from_stage) > STAGES.index(to_stage):
        raise ValueError("from-stage must be earlier than or equal to to-stage")


def resolve_selected_folds(args, config_folds: list[dict[str, object]]) -> list[dict[str, object]]:
    if args.experiment_name in {"expanding_gap", "expanding_nogap", "rolling5_gap", "rolling5_nogap"} and (
        args.generated_first_test_year is not None or args.generated_last_test_year is not None or not config_folds
    ):
        first_year = args.generated_first_test_year or 2020
        last_year = args.generated_last_test_year or 2026
        folds = generate_experiment_folds(
            args.experiment_name,
            train_start=args.generated_train_start,
            first_test_year=first_year,
            last_test_year=last_year,
            last_test_end=args.generated_last_test_end,
        )
    else:
        folds = list(config_folds)
    if args.fold_id is not None:
        folds = [fold for fold in folds if str(fold.get("fold_id")) == args.fold_id]
    return folds


def _annotate_targets(targets: pd.DataFrame, run_id: str, fold_id: str, score_version: str) -> pd.DataFrame:
    if targets.empty:
        return targets
    out = targets.copy()
    out["run_id"] = run_id
    out["fold_id"] = fold_id
    out["score_version"] = score_version
    return out


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
