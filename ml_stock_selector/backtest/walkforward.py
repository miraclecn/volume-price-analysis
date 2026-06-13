from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
import json

import pandas as pd

from ml_stock_selector.backtest.engine import BacktestConfig, BacktestResult, run_holding_aware_backtest
from ml_stock_selector.backtest.execution import ExecutionConfig
from ml_stock_selector.constants import FEATURE_SET_BASELINE_A
from ml_stock_selector.constants import MODEL_TYPE_ACTIVE_RANKER, MODEL_TYPE_RANKER, MODEL_TYPE_RISK
from ml_stock_selector.feature_store_reader import FeatureStoreSpec
from ml_stock_selector.matrix_cache import (
    FoldMatrixCache,
    build_fold_matrix_cache,
    is_fold_manifest_complete,
    mark_fold_manifest_failed,
    read_fold_manifest,
    update_fold_manifest_status,
)
from ml_stock_selector.models.active_ranker import train_active_ranker
from ml_stock_selector.models.alpha_ranker import train_alpha_ranker
from ml_stock_selector.models.artifacts import ModelArtifact
from ml_stock_selector.models.config import artifact_params_json, ranker_config_from_model_section, risk_config_from_model_section
from ml_stock_selector.models.fold_cache_training import train_three_models_from_fold_cache
from ml_stock_selector.models.risk_model import train_risk_model
from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.constructor import construct_portfolio_targets_v2
from ml_stock_selector.portfolio.holding_policy import HoldingPolicy
from ml_stock_selector.prediction import build_three_model_prediction_rows, predict_with_model, write_chunked_fold_predictions
from ml_stock_selector.registry import register_model
from ml_stock_selector.runtime.artifacts import write_model_artifact_bundle, write_walkforward_fold_artifacts
from ml_stock_selector.sample_builder import build_training_samples
from ml_stock_selector.scoring import add_context_score, add_liquidity_score, score_candidates_v2
from ml_stock_selector.universe import apply_universe_filter


@dataclass(frozen=True)
class WalkForwardFoldResult:
    fold_id: str
    model_ids: list[str]
    predictions: pd.DataFrame
    targets: pd.DataFrame
    backtest_result: BacktestResult
    metrics: dict[str, float]


def _between(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)].copy()


def _portfolio_constraints_from_config(config) -> PortfolioConstraints:
    portfolio_root = config.portfolio
    portfolio = portfolio_root.get("v2", portfolio_root)
    ml_v2 = config.ml_v2
    holding = portfolio.get("holding", {}) if isinstance(portfolio.get("holding", {}), dict) else {}
    exit_config = portfolio.get("exit", {}) if isinstance(portfolio.get("exit", {}), dict) else {}
    def v2_threshold(key: str, fallback: object | None = None) -> object:
        if key in ml_v2:
            return ml_v2[key]
        if key in portfolio:
            return portfolio[key]
        return fallback

    return PortfolioConstraints(
        min_trade_score=float(portfolio["min_trade_score"]),
        min_adv20_amount=float(portfolio.get("min_adv20_amount", 0.0)) or None,
        target_positions=int(portfolio["target_positions"]),
        hard_max_positions=int(portfolio["hard_max_positions"]),
        max_industry_names=int(portfolio["max_industry_names"]),
        max_unknown_industry_names=int(portfolio.get("max_unknown_industry_names", 1)),
        max_initial_entries=int(portfolio.get("max_initial_entries", portfolio["target_positions"])),
        max_new_entries_per_day=int(portfolio["max_new_entries_per_day"]),
        allow_cash=bool(portfolio["allow_cash"]),
        min_candidate_pool_size=int(v2_threshold("min_candidate_pool_size", 5)),
        candidate_min_trade_score=float(v2_threshold("candidate_min_trade_score", portfolio["min_trade_score"])),
        candidate_absolute_min_rank_pct=float(v2_threshold("candidate_absolute_min_rank_pct")),
        candidate_active_min_rank_pct=float(v2_threshold("candidate_active_min_rank_pct")),
        candidate_risk_max_rank_pct=float(v2_threshold("candidate_risk_max_rank_pct")),
        core_absolute_min_rank_pct=float(v2_threshold("core_absolute_min_rank_pct")),
        core_active_min_rank_pct=float(v2_threshold("core_active_min_rank_pct")),
        core_risk_max_rank_pct=float(v2_threshold("core_risk_max_rank_pct")),
        core_min_trade_score=float(v2_threshold("core_min_trade_score")),
        exclude_bse=bool(portfolio.get("exclude_bse", config.universe.get("exclude_bse", False))),
        holding_policy=HoldingPolicy(
            min_hold_days=int(holding.get("min_hold_days", 3)),
            target_hold_days=int(holding.get("target_hold_days", 5)),
            max_hold_days=int(holding.get("max_hold_days", 10)),
            sell_score_threshold=float(exit_config.get("sell_score_threshold", 0.45)),
            risk_exit_rank_pct=float(exit_config.get("risk_exit_rank_pct", 0.85)),
            risk_exit_prob=float(exit_config.get("risk_exit_prob", 0.70)),
            sell_if_not_candidate_after_target_days=bool(exit_config.get("sell_if_not_candidate_after_target_days", True)),
            force_exit_after_max_hold_days=bool(exit_config.get("force_exit_after_max_hold_days", True)),
            allow_score_exit_before_min_hold=bool(exit_config.get("allow_score_exit_before_min_hold", False)),
        ),
    )


def run_walkforward_experiment(
    config,
    con,
    normalized_bars: pd.DataFrame,
    feature_mart: pd.DataFrame,
    labels: pd.DataFrame,
    tradeability: pd.DataFrame,
    artifact_dir: Path | str = "outputs/ml/artifacts",
    run_id: str | None = None,
    run_artifact_root: Path | str | None = None,
) -> list[WalkForwardFoldResult]:
    run_id = run_id or f"wf_three_model_v2_{uuid4().hex[:8]}"
    folds = config.split.get("folds", [])
    if not folds:
        return []
    feature_set_id = str(config.features.get("feature_set_id", FEATURE_SET_BASELINE_A))
    if feature_set_id not in set(feature_mart["feature_set_id"]):
        feature_set_id = str(feature_mart["feature_set_id"].iloc[0])
    horizon = int(config.labels.get("main_horizon", 1))
    if horizon not in set(labels["horizon_d"]):
        horizon = int(labels["horizon_d"].iloc[0])
    label_base = str(config.labels.get("label_base", "from_next_open"))
    exclude_bse = bool(config.universe.get("exclude_bse", False))
    ranker_config = ranker_config_from_model_section(config.model)
    risk_config = risk_config_from_model_section(config.model)
    results: list[WalkForwardFoldResult] = []
    for idx, fold in enumerate(folds):
        fold_id = str(fold.get("fold_id", f"fold_{idx+1}"))
        train_start = str(fold["train_start"])
        train_end = str(fold["train_end"])
        test_start = str(fold["test_start"])
        test_end = str(fold["test_end"])

        train_fm = _between(feature_mart, train_start, train_end)
        test_fm = _between(feature_mart, test_start, test_end)
        if train_fm.empty or test_fm.empty:
            continue

        min_adv20_amount = config.portfolio.get("v2", {}).get("min_adv20_amount", config.portfolio.get("min_adv20_amount"))
        abs_samples = build_training_samples(
            train_fm,
            labels,
            feature_set_id,
            horizon,
            label_base,
            "absolute_label",
            exclude_bse=exclude_bse,
            executable_only=True,
            min_adv20_amount=float(min_adv20_amount) if min_adv20_amount is not None else None,
        )
        active_samples = build_training_samples(
            train_fm,
            labels,
            feature_set_id,
            horizon,
            label_base,
            "active_label",
            exclude_bse=exclude_bse,
            executable_only=True,
            min_adv20_amount=float(min_adv20_amount) if min_adv20_amount is not None else None,
        )
        risk_samples = build_training_samples(
            train_fm,
            labels,
            feature_set_id,
            horizon,
            label_base,
            "risk_label",
            exclude_bse=exclude_bse,
            executable_only=True,
            min_adv20_amount=float(min_adv20_amount) if min_adv20_amount is not None else None,
        )
        if abs_samples.empty or active_samples.empty or risk_samples.empty:
            continue

        fold_artifact_root = _fold_artifact_root(run_artifact_root, fold_id)
        abs_artifact = train_alpha_ranker(abs_samples, feature_set_id, "absolute_label", label_base, horizon, artifact_dir, True, train_config=ranker_config)
        active_artifact = train_active_ranker(active_samples, feature_set_id, "active_label", label_base, horizon, artifact_dir, True, train_config=ranker_config)
        risk_artifact = train_risk_model(risk_samples, feature_set_id, "risk_label", label_base, horizon, artifact_dir, True, train_config=risk_config)
        if fold_artifact_root is not None:
            abs_artifact = write_model_artifact_bundle(fold_artifact_root, "absolute_ranker", abs_artifact)
            active_artifact = write_model_artifact_bundle(fold_artifact_root, "active_ranker", active_artifact)
            risk_artifact = write_model_artifact_bundle(fold_artifact_root, "risk_model", risk_artifact)
        register_model(
            con,
            model_id=abs_artifact.model_id,
            model_type=MODEL_TYPE_RANKER,
            feature_set_id=feature_set_id,
            label_name="absolute_label",
            label_base=label_base,
            horizon_d=horizon,
            artifact_uri=str(abs_artifact.artifact_uri),
            feature_schema_uri=str(abs_artifact.feature_schema_uri),
            params_json=artifact_params_json(abs_artifact),
            metrics_json=json.dumps(abs_artifact.metrics),
            notes=f"walkforward:{run_id}:{fold_id}",
            train_start=train_start,
            train_end=train_end,
            valid_start=str(fold.get("valid_start")),
            valid_end=str(fold.get("valid_end")),
            test_start=test_start,
            test_end=test_end,
        )
        register_model(
            con,
            model_id=active_artifact.model_id,
            model_type=MODEL_TYPE_ACTIVE_RANKER,
            feature_set_id=feature_set_id,
            label_name="active_label",
            label_base=label_base,
            horizon_d=horizon,
            artifact_uri=str(active_artifact.artifact_uri),
            feature_schema_uri=str(active_artifact.feature_schema_uri),
            params_json=artifact_params_json(active_artifact),
            metrics_json=json.dumps(active_artifact.metrics),
            notes=f"walkforward:{run_id}:{fold_id}",
            train_start=train_start,
            train_end=train_end,
            valid_start=str(fold.get("valid_start")),
            valid_end=str(fold.get("valid_end")),
            test_start=test_start,
            test_end=test_end,
        )
        register_model(
            con,
            model_id=risk_artifact.model_id,
            model_type=MODEL_TYPE_RISK,
            feature_set_id=feature_set_id,
            label_name="risk_label",
            label_base=label_base,
            horizon_d=horizon,
            artifact_uri=str(risk_artifact.artifact_uri),
            feature_schema_uri=str(risk_artifact.feature_schema_uri),
            params_json=artifact_params_json(risk_artifact),
            metrics_json=json.dumps(risk_artifact.metrics),
            notes=f"walkforward:{run_id}:{fold_id}",
            train_start=train_start,
            train_end=train_end,
            valid_start=str(fold.get("valid_start")),
            valid_end=str(fold.get("valid_end")),
            test_start=test_start,
            test_end=test_end,
        )

        test_fm = apply_universe_filter(test_fm, exclude_bse=exclude_bse)
        predictions = build_three_model_prediction_rows(
            test_fm,
            predict_with_model(test_fm, abs_artifact),
            predict_with_model(test_fm, active_artifact),
            predict_with_model(test_fm, risk_artifact),
            abs_artifact,
            active_artifact,
            risk_artifact,
        )
        enrich_cols = ["trade_date", "code", "industry_code", "industry_name", "is_st", "is_paused", "adv20_amount", "can_buy_next_open", "is_bse"]
        predictions = predictions.merge(test_fm[enrich_cols], on=["trade_date", "code"], how="left")
        predictions = score_candidates_v2(add_liquidity_score(add_context_score(predictions)))
        predictions["run_id"] = run_id
        predictions["fold_id"] = fold_id
        predictions["absolute_model_id"] = abs_artifact.model_id
        predictions["active_model_id"] = active_artifact.model_id
        predictions["risk_model_id"] = risk_artifact.model_id
        constraints = _portfolio_constraints_from_config(config)
        targets = construct_portfolio_targets_v2(predictions, constraints, fold_id)
        targets = allocate_weights(
            targets,
            float(config.portfolio["single_name_min_weight"]),
            float(config.portfolio["single_name_max_weight"]),
            bool(config.portfolio["allow_cash"]),
        )
        bars = _between(normalized_bars, test_start, test_end)
        decision_dates = sorted(predictions["trade_date"].dropna().unique())
        backtest = run_holding_aware_backtest(
            predictions,
            bars,
            constraints,
            BacktestConfig(
                1000000.0,
                fold_id,
                ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0),
                decision_dates=decision_dates,
            ),
            min_weight=float(config.portfolio["single_name_min_weight"]),
            max_weight=float(config.portfolio["single_name_max_weight"]),
            allow_cash=bool(config.portfolio["allow_cash"]),
            run_id=run_id,
            fold_id=fold_id,
            score_version="v2_three_model",
        )
        metrics = {"rows": float(len(predictions)), "run_id": run_id, "score_version": "v2_three_model"}
        if fold_artifact_root is not None:
            write_walkforward_fold_artifacts(
                fold_artifact_root,
                predictions=predictions,
                targets=targets,
                diagnostics=backtest.portfolio_diagnostics,
                orders=backtest.orders,
                positions=backtest.positions,
                nav=backtest.nav,
                metrics=metrics,
                models={"absolute": abs_artifact.model_id, "active": active_artifact.model_id, "risk": risk_artifact.model_id},
            )
        results.append(WalkForwardFoldResult(fold_id, [abs_artifact.model_id, active_artifact.model_id, risk_artifact.model_id], predictions, targets, backtest, metrics))
    return results


def run_walkforward_feature_store_experiment(
    config,
    con,
    normalized_bars: pd.DataFrame,
    *,
    run_id: str,
    feature_store_dir: str,
    feature_store_version: str,
    matrix_cache_dir: str,
    feature_set_id: str,
    horizon_d: int,
    label_base: str,
    score_version: str,
    fold_id: str | None = None,
    artifact_dir: Path | str = "outputs/ml/artifacts",
    run_artifact_root: Path | str | None = None,
    batch_size: int = 50000,
    prediction_chunk_size: int = 50000,
    force: bool = False,
) -> list[WalkForwardFoldResult]:
    selected_folds = _select_folds(config.split.get("folds", []), fold_id)
    spec = FeatureStoreSpec(feature_store_dir, feature_store_version, feature_set_id)
    results: list[WalkForwardFoldResult] = []
    matrix_universe_config = {
        **dict(config.universe),
        "min_adv20_amount": config.portfolio.get("v2", {}).get(
            "min_adv20_amount",
            config.portfolio.get("min_adv20_amount"),
        ),
    }
    for fold in selected_folds:
        current_fold_id = str(fold["fold_id"])
        fold_artifact_root = _fold_artifact_root(run_artifact_root, current_fold_id)
        print(
            "fold_id={fold_id} train={train_start}..{train_end} valid={valid_start}..{valid_end} "
            "test={test_start}..{test_end} feature_store_version={feature_store_version} exclude_bse={exclude_bse}".format(
                fold_id=current_fold_id,
                train_start=fold["train_start"],
                train_end=fold["train_end"],
                valid_start=fold.get("valid_start"),
                valid_end=fold.get("valid_end"),
                test_start=fold["test_start"],
                test_end=fold["test_end"],
                feature_store_version=feature_store_version,
                exclude_bse=bool(config.universe.get("exclude_bse", False)),
            )
        )
        cache = FoldMatrixCache.from_paths(run_id, current_fold_id, matrix_cache_dir)
        try:
            if force or not is_fold_manifest_complete(cache, "matrix_built"):
                cache = build_fold_matrix_cache(
                    con,
                    spec,
                    fold,
                    run_id,
                    feature_set_id,
                    horizon_d,
                    label_base,
                    matrix_universe_config,
                    matrix_cache_dir,
                    batch_size=batch_size,
                )
            manifest = read_fold_manifest(cache.manifest_path)
            print(
                "estimated_rows train={train_rows} valid={valid_rows} test={test_rows}".format(
                    train_rows=manifest["train_rows"],
                    valid_rows=manifest["valid_rows"],
                    test_rows=manifest["test_rows"],
                )
            )
            if not force and is_fold_manifest_complete(cache, "models_trained"):
                artifacts = _artifacts_from_manifest(cache, read_fold_manifest(cache.manifest_path))
            else:
                artifacts = train_three_models_from_fold_cache(cache, config, artifact_dir)
                if fold_artifact_root is not None:
                    artifacts = _bundle_three_model_artifacts(fold_artifact_root, artifacts)
                update_fold_manifest_status(cache.manifest_path, "models_trained", artifacts=_artifact_manifest(artifacts))
            if fold_artifact_root is not None and not str(artifacts.absolute.artifact_dir).startswith(str(fold_artifact_root)):
                artifacts = _bundle_three_model_artifacts(fold_artifact_root, artifacts)
            manifest = read_fold_manifest(cache.manifest_path)
            for artifact in [artifacts.absolute, artifacts.active, artifacts.risk]:
                register_model(
                    con,
                    model_id=artifact.model_id,
                    model_type=artifact.model_type,
                    feature_set_id=artifact.feature_set_id,
                    label_name=artifact.label_name,
                    label_base=artifact.label_base,
                    horizon_d=artifact.horizon_d,
                    artifact_uri=str(artifact.artifact_uri),
                    feature_schema_uri=str(artifact.feature_schema_uri),
                    params_json=artifact_params_json(artifact),
                    metrics_json=json.dumps(artifact.metrics),
                    notes=f"walkforward:{run_id}:{current_fold_id}:feature_store={feature_store_version}",
                    run_id=run_id,
                    fold_id=current_fold_id,
                    feature_store_version=feature_store_version,
                    feature_schema_hash=manifest.get("feature_schema_hash"),
                    train_start=str(fold["train_start"]),
                    train_end=str(fold["train_end"]),
                    valid_start=str(fold.get("valid_start")),
                    valid_end=str(fold.get("valid_end")),
                    test_start=str(fold["test_start"]),
                    test_end=str(fold["test_end"]),
                )
            if force or not is_fold_manifest_complete(cache, "predicted"):
                rows_written = write_chunked_fold_predictions(
                    con,
                    cache,
                    artifacts.absolute,
                    artifacts.active,
                    artifacts.risk,
                    score_version=score_version,
                    chunk_size=prediction_chunk_size,
                )
                update_fold_manifest_status(cache.manifest_path, "predicted", prediction_rows=rows_written)
            predictions = _load_fold_predictions(con, run_id, current_fold_id, score_version)
            metadata = con.execute("select * from read_parquet(?)", [str(cache.metadata_test_path)]).fetchdf()
            enriched = predictions.merge(metadata, on=["trade_date", "code"], how="left")
            constraints = _portfolio_constraints_from_config(config)
            targets = construct_portfolio_targets_v2(enriched, constraints, current_fold_id)
            targets = allocate_weights(
                targets,
                float(config.portfolio["single_name_min_weight"]),
                float(config.portfolio["single_name_max_weight"]),
                bool(config.portfolio["allow_cash"]),
            )
            bars = _between(normalized_bars, str(fold["test_start"]), str(fold["test_end"]))
            decision_dates = sorted(enriched["trade_date"].dropna().unique())
            backtest = run_holding_aware_backtest(
                enriched,
                bars,
                constraints,
                BacktestConfig(
                    1000000.0,
                    current_fold_id,
                    ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0),
                    decision_dates=decision_dates,
                ),
                min_weight=float(config.portfolio["single_name_min_weight"]),
                max_weight=float(config.portfolio["single_name_max_weight"]),
                allow_cash=bool(config.portfolio["allow_cash"]),
                run_id=run_id,
                fold_id=current_fold_id,
                score_version=score_version,
            )
            update_fold_manifest_status(cache.manifest_path, "backtested")
            metrics = {"rows": float(len(predictions)), "run_id": run_id, "score_version": score_version}
            if fold_artifact_root is not None:
                write_walkforward_fold_artifacts(
                    fold_artifact_root,
                    predictions=enriched,
                    targets=targets,
                    diagnostics=backtest.portfolio_diagnostics,
                    orders=backtest.orders,
                    positions=backtest.positions,
                    nav=backtest.nav,
                    metrics=metrics,
                    models={"absolute": artifacts.absolute.model_id, "active": artifacts.active.model_id, "risk": artifacts.risk.model_id},
                )
            results.append(
                WalkForwardFoldResult(
                    current_fold_id,
                    artifacts.model_ids,
                    enriched,
                    targets,
                    backtest,
                    metrics,
                )
            )
        except Exception as exc:
            cache.cache_dir.mkdir(parents=True, exist_ok=True)
            mark_fold_manifest_failed(cache.manifest_path, exc)
            raise
    return results


def _fold_artifact_root(run_artifact_root: Path | str | None, fold_id: str) -> Path | None:
    if run_artifact_root is None:
        return None
    root = Path(run_artifact_root) / "folds" / fold_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _bundle_three_model_artifacts(fold_root: Path, artifacts):
    from ml_stock_selector.models.fold_cache_training import ThreeModelFoldArtifacts

    return ThreeModelFoldArtifacts(
        absolute=write_model_artifact_bundle(fold_root, "absolute_ranker", artifacts.absolute),
        active=write_model_artifact_bundle(fold_root, "active_ranker", artifacts.active),
        risk=write_model_artifact_bundle(fold_root, "risk_model", artifacts.risk),
    )


def _select_folds(folds: list[dict[str, object]], fold_id: str | None) -> list[dict[str, object]]:
    if fold_id is None:
        return folds
    selected = [fold for fold in folds if str(fold.get("fold_id")) == fold_id]
    if not selected:
        raise ValueError(f"Unknown fold_id: {fold_id}")
    return selected


def _load_fold_predictions(con, run_id: str, fold_id: str, score_version: str) -> pd.DataFrame:
    return con.execute(
        """
        select *
        from ml_predictions_daily
        where run_id = ? and fold_id = ? and score_version = ?
        order by trade_date, code
        """,
        [run_id, fold_id, score_version],
    ).fetchdf()


def _artifact_manifest(artifacts) -> dict[str, dict[str, object]]:
    return {
        "absolute": _artifact_payload(artifacts.absolute),
        "active": _artifact_payload(artifacts.active),
        "risk": _artifact_payload(artifacts.risk),
    }


def _artifact_payload(artifact: ModelArtifact) -> dict[str, object]:
    return {
        "model_id": artifact.model_id,
        "model_type": artifact.model_type,
        "feature_set_id": artifact.feature_set_id,
        "label_name": artifact.label_name,
        "label_base": artifact.label_base,
        "horizon_d": artifact.horizon_d,
        "feature_schema_uri": str(artifact.feature_schema_uri),
        "artifact_uri": str(artifact.artifact_uri),
        "artifact_dir": str(artifact.artifact_dir),
        "metrics": artifact.metrics,
    }


def _artifacts_from_manifest(cache: FoldMatrixCache, manifest: dict[str, object]):
    from ml_stock_selector.models.fold_cache_training import ThreeModelFoldArtifacts

    raw = manifest.get("artifacts")
    if not isinstance(raw, dict):
        raise ValueError(f"fold manifest is missing trained model artifacts: {cache.manifest_path}")
    return ThreeModelFoldArtifacts(
        absolute=_artifact_from_payload(raw["absolute"]),
        active=_artifact_from_payload(raw["active"]),
        risk=_artifact_from_payload(raw["risk"]),
    )


def _artifact_from_payload(payload: dict[str, object]) -> ModelArtifact:
    artifact_uri = Path(str(payload["artifact_uri"]))
    return ModelArtifact(
        model_id=str(payload["model_id"]),
        model_type=str(payload["model_type"]),
        feature_set_id=str(payload["feature_set_id"]),
        label_name=str(payload["label_name"]),
        label_base=str(payload["label_base"]),
        horizon_d=int(payload["horizon_d"]),
        feature_schema_uri=Path(str(payload["feature_schema_uri"])),
        artifact_uri=artifact_uri,
        artifact_dir=Path(str(payload.get("artifact_dir", artifact_uri.parent))),
        metrics=dict(payload.get("metrics", {})),
    )
