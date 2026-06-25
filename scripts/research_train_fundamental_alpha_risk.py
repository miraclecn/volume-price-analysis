from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_stock_selector.config import load_ml_config
from ml_stock_selector.feature_matrix import FeatureSchema, load_feature_schema, save_feature_schema
from ml_stock_selector.fundamental_features import FUNDAMENTAL_FEATURE_COLUMNS, load_fundamental_features_for_metadata
from ml_stock_selector.matrix_cache import (
    FoldMatrixCache,
    is_fold_manifest_complete,
    load_cached_matrix,
    read_fold_manifest,
    update_fold_manifest_status,
)
from ml_stock_selector.models.artifacts import ModelArtifact
from ml_stock_selector.models.fold_cache_training import AlphaRiskFoldArtifacts, train_alpha_risk_models_from_fold_cache
from ml_stock_selector.prediction import write_chunked_alpha_risk_fold_predictions
from ml_stock_selector.storage import init_ml_db


DEFAULT_RUN_ID = "wf_v2_ret5_fundamental_train_20260620"
DEFAULT_SCORE_VERSION = "v2_alpha_ret5d_fundamental_train_20260620"
DEFAULT_SOURCE_RUN_ID = "wf_v2_ret5_alpha_risk_20260619"
DEFAULT_SOURCE_CACHE_ROOT = "outputs/ml/cache/folds_ret5_alpha_risk_20260619"
DEFAULT_TARGET_CACHE_ROOT = "outputs/ml/cache/folds_ret5_fundamental_20260620"
DEFAULT_ARTIFACT_DIR = "outputs/ml/artifacts/ret5_fundamental_train_20260620"
DEFAULT_REPORT_DIR = "outputs/ml/reports/ret5_fundamental_train_replay_2020_2025"
DEFAULT_ML_DB = "outputs/ml/ml_ret5_alpha_risk_20260619.duckdb"
DEFAULT_RAW_DB = "/home/nan/alpha-data-local/output/raw.duckdb"
DEFAULT_CONFIG = "config/ml_walkforward_adv10m_ret5_alpha_risk.toml"
STAGE_ORDER = {"matrix": 0, "train": 1, "predict": 2, "replay": 3}
TRAIN_WINDOW_MODES = {"source_gap", "expanding_no_gap", "rolling_5y_no_gap", "rolling_3y_no_gap"}


@dataclass(frozen=True)
class SelectedFold:
    fold_id: str
    year: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train alpha+risk ret5 models with raw daily fundamental features appended to the existing matrix cache."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ml-db", default=DEFAULT_ML_DB)
    parser.add_argument("--raw-db", default=DEFAULT_RAW_DB)
    parser.add_argument("--source-cache-root", default=DEFAULT_SOURCE_CACHE_ROOT)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--target-cache-root", default=DEFAULT_TARGET_CACHE_ROOT)
    parser.add_argument("--target-run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--score-version", default=DEFAULT_SCORE_VERSION)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--fold-id", action="append", help="Fold id to run, e.g. wf_2020. Repeat for multiple folds.")
    parser.add_argument("--years", nargs="+", type=int, help="Replay years. Defaults to selected folds or 2020-2025.")
    parser.add_argument("--to-stage", choices=sorted(STAGE_ORDER), default="replay")
    parser.add_argument("--source-cache-kind", choices=["raw", "augmented"], default="raw")
    parser.add_argument("--train-window-mode", choices=sorted(TRAIN_WINDOW_MODES), default="source_gap")
    parser.add_argument("--n-estimators", type=int, help="Override LightGBM max boosting rounds for both alpha and risk.")
    parser.add_argument("--early-stopping-rounds", type=int, help="Override LightGBM early stopping patience for both alpha and risk.")
    parser.add_argument("--alpha-early-stop-target", choices=["default", "future_max_gain"], default="default")
    parser.add_argument("--force", action="store_true", help="Rebuild/retrain/repredict even if target artifacts exist.")
    parser.add_argument("--matrix-chunk-rows", type=int, default=200_000)
    parser.add_argument("--prediction-chunk-size", type=int, default=50_000)
    parser.add_argument("--variants", nargs="*", default=["base"], help="Replay variants; use 'all' for every risk-control variant.")
    parser.add_argument("--base-portfolio-id", default="fundamental_train_replay")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_ml_config(args.config)
    config = apply_training_overrides(config, n_estimators=args.n_estimators, early_stopping_rounds=args.early_stopping_rounds)
    selected = _select_folds(config.split["folds"], args.fold_id)
    years = args.years if args.years else [fold.year for fold in selected]
    target_stage = STAGE_ORDER[args.to_stage]
    if args.source_cache_kind == "raw" and args.train_window_mode != "source_gap":
        raise ValueError("non-source_gap train windows require --source-cache-kind augmented")

    con = None
    try:
        for fold in selected:
            source_cache = FoldMatrixCache.from_paths(args.source_run_id, fold.fold_id, args.source_cache_root)
            target_cache = FoldMatrixCache.from_paths(args.target_run_id, fold.fold_id, args.target_cache_root)
            if target_stage >= STAGE_ORDER["matrix"]:
                if args.source_cache_kind == "augmented":
                    build_split_variant_fold_cache(
                        source_cache,
                        target_cache,
                        train_window_mode=args.train_window_mode,
                        force=args.force,
                        chunk_rows=args.matrix_chunk_rows,
                    )
                else:
                    build_augmented_fold_cache(
                        source_cache,
                        target_cache,
                        raw_db_path=Path(args.raw_db),
                        force=args.force,
                        chunk_rows=args.matrix_chunk_rows,
                    )
            artifacts: AlphaRiskFoldArtifacts | None = None
            if target_stage >= STAGE_ORDER["train"]:
                materialize_alpha_early_stop_target(
                    target_cache,
                    ml_db_path=Path(args.ml_db),
                    target_name=args.alpha_early_stop_target,
                    force=args.force,
                )
                artifacts = train_or_load_fold_models(
                    target_cache,
                    config,
                    artifact_root=Path(args.artifact_dir),
                    force=args.force,
                )
            if target_stage >= STAGE_ORDER["predict"]:
                if con is None:
                    con = init_ml_db(args.ml_db)
                artifacts = artifacts or load_alpha_risk_artifacts_from_manifest(target_cache)
                if args.force or not is_fold_manifest_complete(target_cache, "predicted"):
                    con.execute(
                        """
                        delete from ml_prediction_raw_daily
                        where run_id = ? and fold_id = ? and score_version = ?
                        """,
                        [target_cache.run_id, target_cache.fold_id, args.score_version],
                    )
                    con.execute(
                        """
                        delete from ml_predictions_daily
                        where run_id = ? and fold_id = ? and score_version = ?
                        """,
                        [target_cache.run_id, target_cache.fold_id, args.score_version],
                    )
                    rows = write_chunked_alpha_risk_fold_predictions(
                        con,
                        target_cache,
                        artifacts.absolute,
                        artifacts.risk,
                        score_version=args.score_version,
                        chunk_size=args.prediction_chunk_size,
                    )
                    update_fold_manifest_status(target_cache.manifest_path, "predicted", prediction_rows=rows)
                    print(f"predicted fold={fold.fold_id} rows={rows}", flush=True)
                else:
                    print(f"skip prediction fold={fold.fold_id}: target manifest already predicted", flush=True)
                if con is not None:
                    con.close()
                    con = None
    finally:
        if con is not None:
            con.close()

    if target_stage >= STAGE_ORDER["replay"]:
        run_live_like_replay(
            ml_db=Path(args.ml_db),
            out_dir=Path(args.report_dir),
            run_id=args.target_run_id,
            score_version=args.score_version,
            base_portfolio_id=args.base_portfolio_id,
            years=years,
            variant_names=args.variants,
        )
        write_prediction_monotonicity_report(
            ml_db=Path(args.ml_db),
            out_dir=Path(args.report_dir),
            run_id=args.target_run_id,
            score_version=args.score_version,
            years=years,
            horizon_d=int(config.labels["main_horizon"]),
            label_base=str(config.labels["label_base"]),
        )


def build_augmented_fold_cache(
    source_cache: FoldMatrixCache,
    target_cache: FoldMatrixCache,
    *,
    raw_db_path: Path,
    force: bool,
    chunk_rows: int,
) -> None:
    if not force and is_fold_manifest_complete(target_cache, "matrix_built"):
        print(f"skip matrix fold={target_cache.fold_id}: target cache already built", flush=True)
        return
    _ensure_source_cache(source_cache)
    target_cache.cache_dir.mkdir(parents=True, exist_ok=True)
    _link_static_cache_files(source_cache, target_cache, force=force)

    for split_name, source_matrix, source_sparse, source_metadata, target_matrix, target_sparse in [
        ("train", source_cache.x_train_dense_path, source_cache.x_train_path, source_cache.metadata_train_path, target_cache.x_train_dense_path, target_cache.x_train_path),
        ("valid", source_cache.x_valid_dense_path, source_cache.x_valid_path, source_cache.metadata_valid_path, target_cache.x_valid_dense_path, target_cache.x_valid_path),
        ("test", source_cache.x_test_dense_path, source_cache.x_test_path, source_cache.metadata_test_path, target_cache.x_test_dense_path, target_cache.x_test_path),
    ]:
        if target_matrix.exists() and not force:
            print(f"skip matrix split fold={target_cache.fold_id} split={split_name}: {target_matrix}", flush=True)
            continue
        _write_augmented_matrix(
            source_dense_path=source_matrix,
            source_sparse_path=source_sparse,
            metadata_path=source_metadata,
            target_dense_path=target_matrix,
            target_sparse_path=target_sparse,
            raw_db_path=raw_db_path,
            chunk_rows=chunk_rows,
            split_name=split_name,
            fold_id=target_cache.fold_id,
        )

    schema = _augmented_schema(source_cache.feature_schema_path)
    save_feature_schema(schema, target_cache.feature_schema_path)
    source_manifest = _matrix_manifest_base(read_fold_manifest(source_cache.manifest_path))
    _write_json(
        target_cache.manifest_path,
        {
            **source_manifest,
            "run_id": target_cache.run_id,
            "fold_id": target_cache.fold_id,
            "status": "matrix_built",
            "source_run_id": source_cache.run_id,
            "source_cache_dir": str(source_cache.cache_dir),
            "fundamental_raw_db": str(raw_db_path),
            "fundamental_columns": [target for _, target in FUNDAMENTAL_FEATURE_COLUMNS],
            "feature_set_id": schema.feature_set_id,
            "feature_schema_hash": None,
            "schema_version": schema.schema_version,
            "matrix_format": "dense_float32_npy",
            "matrix_built_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"built matrix fold={target_cache.fold_id} dir={target_cache.cache_dir}", flush=True)


def build_split_variant_fold_cache(
    source_cache: FoldMatrixCache,
    target_cache: FoldMatrixCache,
    *,
    train_window_mode: str,
    force: bool,
    chunk_rows: int,
) -> None:
    if not force and is_fold_manifest_complete(target_cache, "matrix_built"):
        print(f"skip matrix fold={target_cache.fold_id}: target cache already built", flush=True)
        return
    _ensure_source_cache(source_cache)
    target_cache.cache_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = _matrix_manifest_base(read_fold_manifest(source_cache.manifest_path))
    train_start, train_end = train_window_for_mode(source_manifest, train_window_mode)
    if train_window_mode == "source_gap":
        _link_augmented_all_files(source_cache, target_cache, force=force)
        train_rows = int(source_manifest["train_rows"])
    else:
        _link_augmented_static_files(source_cache, target_cache, force=force)
        train_rows = _write_train_window_from_augmented_source(
            source_cache,
            target_cache,
            train_start=train_start,
            train_end=train_end,
            chunk_rows=chunk_rows,
        )
    source_schema = load_feature_schema(source_cache.feature_schema_path)
    save_feature_schema(source_schema, target_cache.feature_schema_path)
    _write_json(
        target_cache.manifest_path,
        {
            **source_manifest,
            "run_id": target_cache.run_id,
            "fold_id": target_cache.fold_id,
            "status": "matrix_built",
            "source_run_id": source_cache.run_id,
            "source_cache_dir": str(source_cache.cache_dir),
            "source_cache_kind": "augmented",
            "train_window_mode": train_window_mode,
            "train_start": train_start,
            "train_end": train_end,
            "train_rows": train_rows,
            "feature_set_id": source_schema.feature_set_id,
            "schema_version": source_schema.schema_version,
            "feature_schema_hash": None,
            "matrix_format": "dense_float32_npy",
            "matrix_built_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(
        f"built split matrix fold={target_cache.fold_id} mode={train_window_mode} "
        f"train={train_start}..{train_end} rows={train_rows}",
        flush=True,
    )


def train_window_for_mode(source_manifest: dict[str, object], train_window_mode: str) -> tuple[str, str]:
    if train_window_mode not in TRAIN_WINDOW_MODES:
        raise ValueError(f"unknown train_window_mode: {train_window_mode}")
    train_start = str(source_manifest["train_start"])
    train_end = str(source_manifest["train_end"])
    valid_end = str(source_manifest["valid_end"])
    if train_window_mode == "source_gap":
        return train_start, train_end
    if train_window_mode == "expanding_no_gap":
        return train_start, valid_end
    window_years = _rolling_years_for_mode(train_window_mode)
    valid_end_year = int(valid_end[:4])
    candidate_start = f"{valid_end_year - window_years + 1:04d}-01-01"
    return max(train_start, candidate_start), valid_end


def apply_training_overrides(config, *, n_estimators: int | None, early_stopping_rounds: int | None):
    if n_estimators is None and early_stopping_rounds is None:
        return config
    runtime = dict(config.model.get("lightgbm_runtime", {}))
    if n_estimators is not None:
        runtime["n_estimators"] = int(n_estimators)
    if early_stopping_rounds is not None:
        runtime["early_stopping_rounds"] = int(early_stopping_rounds)
    model = dict(config.model)
    model["lightgbm_runtime"] = runtime
    return replace(config, model=model)


def materialize_alpha_early_stop_target(
    cache: FoldMatrixCache,
    *,
    ml_db_path: Path,
    target_name: str,
    force: bool,
) -> None:
    if target_name == "default":
        return
    if target_name != "future_max_gain":
        raise ValueError(f"unknown alpha early-stop target: {target_name}")
    manifest = read_fold_manifest(cache.manifest_path)
    output_name = "y_alpha_eval_valid_future_max_gain.npy"
    output_path = cache.cache_dir / output_name
    if output_path.exists() and not force and manifest.get("alpha_eval_target") == target_name:
        return
    metadata = _read_metadata(cache.metadata_valid_path)
    labels = _load_label_values_for_metadata(
        ml_db_path,
        metadata,
        value_column="future_max_gain",
        horizon_d=int(manifest["horizon_d"]),
        label_base=str(manifest["label_base"]),
    )
    np.save(output_path, labels.astype(np.float32, copy=False))
    manifest.update(
        {
            "alpha_eval_target": target_name,
            "alpha_eval_metric": "future_max_gain_rank_ic",
            "alpha_eval_valid_path": output_name,
            "alpha_eval_missing_count": int(np.isnan(labels).sum()),
            "alpha_eval_materialized_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_json(cache.manifest_path, manifest)


def train_or_load_fold_models(
    cache: FoldMatrixCache,
    config,
    *,
    artifact_root: Path,
    force: bool,
) -> AlphaRiskFoldArtifacts:
    if not force:
        try:
            artifacts = load_alpha_risk_artifacts_from_manifest(cache)
            print(f"skip train fold={cache.fold_id}: target manifest already has artifacts", flush=True)
            return artifacts
        except (FileNotFoundError, KeyError, ValueError):
            pass
    fold_artifact_dir = artifact_root / cache.fold_id
    artifacts = train_alpha_risk_models_from_fold_cache(cache, config, fold_artifact_dir)
    update_fold_manifest_status(cache.manifest_path, "models_trained", artifacts=_artifact_manifest(artifacts), model_mode="alpha_risk")
    print(f"trained fold={cache.fold_id} models={artifacts.model_ids}", flush=True)
    return artifacts


def load_alpha_risk_artifacts_from_manifest(cache: FoldMatrixCache) -> AlphaRiskFoldArtifacts:
    manifest = read_fold_manifest(cache.manifest_path)
    raw = manifest.get("artifacts")
    if not isinstance(raw, dict) or "absolute" not in raw or "risk" not in raw:
        raise ValueError(f"fold manifest is missing alpha/risk artifacts: {cache.manifest_path}")
    artifacts = AlphaRiskFoldArtifacts(
        absolute=_artifact_from_payload(raw["absolute"]),
        risk=_artifact_from_payload(raw["risk"]),
    )
    expected_schema = cache.feature_schema_path.resolve()
    for artifact in [artifacts.absolute, artifacts.risk]:
        if artifact.feature_schema_uri.resolve() != expected_schema:
            raise ValueError(
                f"artifact schema mismatch for {cache.fold_id}: "
                f"{artifact.feature_schema_uri} != {cache.feature_schema_path}"
            )
    if artifacts.absolute.feature_set_id != str(manifest.get("feature_set_id")):
        raise ValueError(f"artifact feature_set_id mismatch for {cache.fold_id}")
    return artifacts


def run_live_like_replay(
    *,
    ml_db: Path,
    out_dir: Path,
    run_id: str,
    score_version: str,
    base_portfolio_id: str,
    years: list[int],
    variant_names: list[str],
) -> None:
    import scripts.research_risk_controls as risk_controls

    risk_controls._set_run_context(run_id, score_version, base_portfolio_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = risk_controls._variants()
    if "all" not in set(variant_names):
        wanted = set(variant_names)
        variants = [variant for variant in variants if variant.name in wanted]
        missing = wanted - {variant.name for variant in variants}
        if missing:
            raise ValueError(f"unknown replay variants: {sorted(missing)}")
    con = duckdb.connect(str(ml_db), read_only=True)
    try:
        market_state = risk_controls._load_market_state(con, min(years), max(years))
        all_results = []
        for year in years:
            scored = risk_controls._load_scored_candidates(con, year)
            bars = risk_controls._load_bars(con, year)
            for variant in variants:
                print(f"replay year={year} variant={variant.name}", flush=True)
                all_results.append(risk_controls.run_research_backtest(scored, bars, market_state, variant, year))
    finally:
        con.close()

    summary = risk_controls._summarize_results(all_results)
    yearly = risk_controls._summarize_years(all_results)
    orders = risk_controls._concat_result_frames(all_results, "orders")
    nav = risk_controls._concat_result_frames(all_results, "nav")
    diagnostics = risk_controls._concat_result_frames(all_results, "diagnostics")
    summary.to_csv(out_dir / "risk_control_summary.csv", index=False)
    yearly.to_csv(out_dir / "risk_control_yearly.csv", index=False)
    orders.to_csv(out_dir / "risk_control_orders.csv", index=False)
    nav.to_csv(out_dir / "risk_control_nav.csv", index=False)
    diagnostics.to_csv(out_dir / "risk_control_diagnostics.csv", index=False)
    if not summary.empty:
        print(summary.sort_values(["loss_to_win", "annual_return"], ascending=[True, False]).to_string(index=False), flush=True)
    print(f"wrote replay reports: {out_dir}", flush=True)


def write_prediction_monotonicity_report(
    *,
    ml_db: Path,
    out_dir: Path,
    run_id: str,
    score_version: str,
    years: list[int],
    horizon_d: int,
    label_base: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    start = f"{min(years)}-01-01"
    end = f"{max(years)}-12-31"
    con = duckdb.connect(str(ml_db), read_only=True)
    try:
        deciles = con.execute(
            """
            with joined as (
                select
                    p.trade_date,
                    cast(substr(p.trade_date, 1, 4) as integer) as year,
                    p.code,
                    p.alpha_score,
                    p.alpha_rank_pct,
                    l.future_max_gain,
                    l.future_ret,
                    ntile(10) over (partition by p.trade_date order by p.alpha_score) as score_decile
                from ml_predictions_daily p
                join ml_labels_daily l
                  on p.trade_date = l.trade_date
                 and p.code = l.code
                 and p.horizon_d = l.horizon_d
                where p.run_id = ?
                  and p.score_version = ?
                  and p.trade_date between ? and ?
                  and l.horizon_d = ?
                  and l.label_base = ?
            )
            select
                year,
                score_decile,
                count(*) as row_count,
                avg(alpha_rank_pct) as mean_alpha_rank_pct,
                avg(future_max_gain) as mean_future_max_gain,
                avg(future_ret) as mean_future_ret,
                avg(case when future_ret > 0 then 1.0 else 0.0 end) as positive_ret_rate
            from joined
            group by year, score_decile
            order by year, score_decile
            """,
            [run_id, score_version, start, end, int(horizon_d), str(label_base)],
        ).fetchdf()
    finally:
        con.close()
    deciles.to_csv(out_dir / "prediction_monotonicity_deciles.csv", index=False)
    summary = _monotonicity_summary(deciles)
    summary.to_csv(out_dir / "prediction_monotonicity_summary.csv", index=False)
    if not summary.empty:
        print(summary.to_string(index=False), flush=True)


def _monotonicity_summary(deciles: pd.DataFrame) -> pd.DataFrame:
    if deciles.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "bucket_count",
                "future_max_gain_violation_count",
                "future_ret_violation_count",
                "future_max_gain_monotonic",
                "future_ret_monotonic",
                "future_max_gain_decile_corr",
                "future_ret_decile_corr",
            ]
        )
    frames = [deciles.copy()]
    all_frame = deciles.groupby("score_decile", as_index=False).agg(
        row_count=("row_count", "sum"),
        mean_alpha_rank_pct=("mean_alpha_rank_pct", "mean"),
        mean_future_max_gain=("mean_future_max_gain", "mean"),
        mean_future_ret=("mean_future_ret", "mean"),
        positive_ret_rate=("positive_ret_rate", "mean"),
    )
    all_frame["year"] = "ALL"
    frames.append(all_frame[deciles.columns])
    rows = []
    for year, group in pd.concat(frames, ignore_index=True).groupby("year", sort=False):
        ordered = group.sort_values("score_decile").reset_index(drop=True)
        gain_diffs = ordered["mean_future_max_gain"].diff().dropna()
        ret_diffs = ordered["mean_future_ret"].diff().dropna()
        gain_violations = int((gain_diffs < -1e-12).sum())
        ret_violations = int((ret_diffs < -1e-12).sum())
        rows.append(
            {
                "year": year,
                "bucket_count": int(len(ordered)),
                "future_max_gain_violation_count": gain_violations,
                "future_ret_violation_count": ret_violations,
                "future_max_gain_monotonic": gain_violations == 0,
                "future_ret_monotonic": ret_violations == 0,
                "future_max_gain_decile_corr": float(ordered["score_decile"].corr(ordered["mean_future_max_gain"])),
                "future_ret_decile_corr": float(ordered["score_decile"].corr(ordered["mean_future_ret"])),
            }
        )
    return pd.DataFrame(rows)


def _write_augmented_matrix(
    *,
    source_dense_path: Path,
    source_sparse_path: Path,
    metadata_path: Path,
    target_dense_path: Path,
    target_sparse_path: Path,
    raw_db_path: Path,
    chunk_rows: int,
    split_name: str,
    fold_id: str,
) -> None:
    source_matrix = load_cached_matrix(source_dense_path, source_sparse_path)
    metadata = _read_metadata(metadata_path)
    rows = int(source_matrix.shape[0])
    if len(metadata) != rows:
        raise ValueError(f"metadata row count mismatch for {fold_id}/{split_name}: metadata={len(metadata)} matrix={rows}")

    base_cols = int(source_matrix.shape[1])
    fundamental_cols = [target for _, target in FUNDAMENTAL_FEATURE_COLUMNS]
    target_cols = base_cols + len(fundamental_cols)
    target_dense_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_dense_path.with_name(f".{target_dense_path.stem}.{uuid4().hex}.npy")
    if target_sparse_path.exists():
        target_sparse_path.unlink()
    try:
        out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(rows, target_cols))
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            base_chunk = source_matrix[start:end]
            if sparse.issparse(base_chunk):
                base_values = base_chunk.toarray().astype(np.float32, copy=False)
            else:
                base_values = np.asarray(base_chunk, dtype=np.float32)
            fundamental = load_fundamental_features_for_metadata(raw_db_path, metadata.iloc[start:end])
            out[start:end, :base_cols] = base_values
            out[start:end, base_cols:] = fundamental.to_numpy(dtype=np.float32, copy=False)
            print(f"matrix fold={fold_id} split={split_name} rows={end}/{rows}", flush=True)
        del out
        os.replace(tmp_path, target_dense_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_train_window_from_augmented_source(
    source_cache: FoldMatrixCache,
    target_cache: FoldMatrixCache,
    *,
    train_start: str,
    train_end: str,
    chunk_rows: int,
) -> int:
    sources = [
        (
            "source_train",
            source_cache.x_train_dense_path,
            source_cache.x_train_path,
            source_cache.metadata_train_path,
            source_cache.y_abs_train_path,
            source_cache.y_active_train_path,
            source_cache.y_risk_train_path,
        ),
        (
            "source_valid",
            source_cache.x_valid_dense_path,
            source_cache.x_valid_path,
            source_cache.metadata_valid_path,
            source_cache.y_abs_valid_path,
            source_cache.y_active_valid_path,
            source_cache.y_risk_valid_path,
        ),
    ]
    selected_parts = []
    total_rows = 0
    feature_cols = None
    for name, dense_path, sparse_path, metadata_path, y_abs_path, y_active_path, y_risk_path in sources:
        matrix = load_cached_matrix(dense_path, sparse_path)
        metadata = _read_metadata(metadata_path)
        if len(metadata) != int(matrix.shape[0]):
            raise ValueError(f"metadata row count mismatch for {target_cache.fold_id}/{name}")
        dates = metadata["trade_date"].astype(str)
        mask = ((dates >= train_start) & (dates <= train_end)).to_numpy()
        rows = int(mask.sum())
        total_rows += rows
        feature_cols = int(matrix.shape[1])
        selected_parts.append(
            {
                "name": name,
                "matrix": matrix,
                "metadata": metadata,
                "mask": mask,
                "y_abs_path": y_abs_path,
                "y_active_path": y_active_path,
                "y_risk_path": y_risk_path,
                "rows": rows,
            }
        )
    if feature_cols is None:
        raise ValueError("no source matrices found")
    if total_rows <= 0:
        raise ValueError(f"empty train window for {target_cache.fold_id}: {train_start}..{train_end}")

    target_cache.x_train_path.unlink(missing_ok=True)
    tmp_path = target_cache.x_train_dense_path.with_name(f".{target_cache.x_train_dense_path.stem}.{uuid4().hex}.npy")
    target_cache.x_train_dense_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_frames: list[pd.DataFrame] = []
    abs_parts = []
    active_parts = []
    risk_parts = []
    try:
        out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(total_rows, feature_cols))
        offset = 0
        for part in selected_parts:
            if part["rows"] <= 0:
                continue
            matrix = part["matrix"]
            mask = part["mask"]
            metadata = part["metadata"]
            metadata_frames.append(metadata.loc[mask].copy())
            abs_parts.append(np.load(part["y_abs_path"], mmap_mode="r")[mask].astype(np.float32, copy=False))
            active_parts.append(np.load(part["y_active_path"], mmap_mode="r")[mask].astype(np.float32, copy=False))
            risk_parts.append(np.load(part["y_risk_path"], mmap_mode="r")[mask].astype(np.float32, copy=False))
            for start in range(0, len(mask), chunk_rows):
                end = min(start + chunk_rows, len(mask))
                chunk_mask = mask[start:end]
                if not chunk_mask.any():
                    continue
                chunk = matrix[start:end]
                values = chunk.toarray().astype(np.float32, copy=False) if sparse.issparse(chunk) else np.asarray(chunk, dtype=np.float32)
                selected = values[chunk_mask]
                rows = int(selected.shape[0])
                out[offset : offset + rows] = selected
                offset += rows
                print(
                    f"matrix fold={target_cache.fold_id} split=train_window part={part['name']} rows={offset}/{total_rows}",
                    flush=True,
                )
        del out
        os.replace(tmp_path, target_cache.x_train_dense_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    metadata_out = pd.concat(metadata_frames, ignore_index=True)
    _write_metadata(target_cache.metadata_train_path, metadata_out)
    np.save(target_cache.y_abs_train_path, np.concatenate(abs_parts).astype(np.float32, copy=False))
    np.save(target_cache.y_active_train_path, np.concatenate(active_parts).astype(np.float32, copy=False))
    np.save(target_cache.y_risk_train_path, np.concatenate(risk_parts).astype(np.float32, copy=False))
    group = metadata_out.groupby("trade_date", sort=False).size().to_numpy(dtype=np.int32)
    np.save(target_cache.group_train_path, group)
    return total_rows


def _augmented_schema(source_schema_path: Path) -> FeatureSchema:
    source = load_feature_schema(source_schema_path)
    fundamental_cols = [target for _, target in FUNDAMENTAL_FEATURE_COLUMNS]
    numeric_columns = list(source.numeric_columns)
    output_columns = list(source.output_columns)
    for column in fundamental_cols:
        if column not in numeric_columns:
            numeric_columns.append(column)
        if column not in output_columns:
            output_columns.append(column)
    fill_values = dict(source.fill_values)
    fill_values.update({column: 0.0 for column in fundamental_cols})
    return FeatureSchema(
        feature_set_id=f"{source.feature_set_id}_fundamental_v1",
        numeric_columns=numeric_columns,
        categorical_columns=list(source.categorical_columns),
        output_columns=output_columns,
        category_levels=dict(source.category_levels),
        fill_values=fill_values,
        schema_version=f"{source.schema_version}_fundamental_v1",
    )


def _link_static_cache_files(source: FoldMatrixCache, target: FoldMatrixCache, *, force: bool) -> None:
    pairs = [
        (source.y_abs_train_path, target.y_abs_train_path),
        (source.y_abs_valid_path, target.y_abs_valid_path),
        (source.y_active_train_path, target.y_active_train_path),
        (source.y_active_valid_path, target.y_active_valid_path),
        (source.y_risk_train_path, target.y_risk_train_path),
        (source.y_risk_valid_path, target.y_risk_valid_path),
        (source.group_train_path, target.group_train_path),
        (source.group_valid_path, target.group_valid_path),
        (source.metadata_train_path, target.metadata_train_path),
        (source.metadata_valid_path, target.metadata_valid_path),
        (source.metadata_test_path, target.metadata_test_path),
    ]
    for source_path, target_path in pairs:
        _link_or_copy(source_path, target_path, force=force)


def _link_augmented_static_files(source: FoldMatrixCache, target: FoldMatrixCache, *, force: bool) -> None:
    pairs = [
        (source.x_valid_dense_path, target.x_valid_dense_path),
        (source.x_test_dense_path, target.x_test_dense_path),
        (source.y_abs_valid_path, target.y_abs_valid_path),
        (source.y_active_valid_path, target.y_active_valid_path),
        (source.y_risk_valid_path, target.y_risk_valid_path),
        (source.group_valid_path, target.group_valid_path),
        (source.metadata_valid_path, target.metadata_valid_path),
        (source.metadata_test_path, target.metadata_test_path),
        (source.feature_schema_path, target.feature_schema_path),
    ]
    for source_path, target_path in pairs:
        _link_or_copy(source_path, target_path, force=force)
    for sparse_path in [target.x_valid_path, target.x_test_path]:
        sparse_path.unlink(missing_ok=True)


def _link_augmented_all_files(source: FoldMatrixCache, target: FoldMatrixCache, *, force: bool) -> None:
    _link_augmented_static_files(source, target, force=force)
    pairs = [
        (source.x_train_dense_path, target.x_train_dense_path),
        (source.y_abs_train_path, target.y_abs_train_path),
        (source.y_active_train_path, target.y_active_train_path),
        (source.y_risk_train_path, target.y_risk_train_path),
        (source.group_train_path, target.group_train_path),
        (source.metadata_train_path, target.metadata_train_path),
    ]
    for source_path, target_path in pairs:
        _link_or_copy(source_path, target_path, force=force)
    target.x_train_path.unlink(missing_ok=True)


def _link_or_copy(source_path: Path, target_path: Path, *, force: bool) -> None:
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        if not force:
            return
        target_path.unlink()
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def _ensure_source_cache(cache: FoldMatrixCache) -> None:
    missing = [
        path
        for path in [
            cache.metadata_train_path,
            cache.metadata_valid_path,
            cache.metadata_test_path,
            cache.y_abs_train_path,
            cache.y_abs_valid_path,
            cache.y_active_train_path,
            cache.y_active_valid_path,
            cache.y_risk_train_path,
            cache.y_risk_valid_path,
            cache.group_train_path,
            cache.group_valid_path,
            cache.feature_schema_path,
            cache.manifest_path,
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"source cache is incomplete for {cache.fold_id}: {missing}")
    for dense, sparse_path in [
        (cache.x_train_dense_path, cache.x_train_path),
        (cache.x_valid_dense_path, cache.x_valid_path),
        (cache.x_test_dense_path, cache.x_test_path),
    ]:
        if not dense.exists() and not sparse_path.exists():
            raise FileNotFoundError(f"source cache missing matrix: {dense} or {sparse_path}")


def _read_metadata(path: Path) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        return con.execute("select * from read_parquet(?) order by trade_date, code", [str(path)]).fetchdf()
    finally:
        con.close()


def _load_label_values_for_metadata(
    ml_db_path: Path,
    metadata: pd.DataFrame,
    *,
    value_column: str,
    horizon_d: int,
    label_base: str,
) -> np.ndarray:
    allowed_columns = {"future_max_gain", "future_ret", "future_max_drawdown"}
    if value_column not in allowed_columns:
        raise ValueError(f"unsupported label value column: {value_column}")
    meta = pd.DataFrame(
        {
            "_row_id": np.arange(len(metadata), dtype=np.int64),
            "trade_date": metadata["trade_date"].astype(str).to_numpy(),
            "code": metadata["code"].astype(str).to_numpy(),
        }
    )
    con = duckdb.connect(str(ml_db_path), read_only=True)
    try:
        con.register("_label_metadata", meta)
        frame = con.execute(
            f"""
            select
                m._row_id,
                try_cast(l.{value_column} as double) as value
            from _label_metadata m
            left join ml_labels_daily l
              on m.trade_date = l.trade_date
             and m.code = l.code
             and l.horizon_d = ?
             and l.label_base = ?
            order by m._row_id
            """,
            [int(horizon_d), str(label_base)],
        ).fetchdf()
    finally:
        con.close()
    return pd.to_numeric(frame["value"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)


def _write_metadata(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    try:
        con.register("_metadata_out", frame)
        con.execute("copy _metadata_out to ? (format parquet, compression zstd)", [str(path)])
    finally:
        con.close()


def _select_folds(raw_folds: list[dict[str, object]], requested: list[str] | None) -> list[SelectedFold]:
    wanted = set(requested or [])
    selected = []
    for fold in raw_folds:
        fold_id = str(fold["fold_id"])
        if wanted and fold_id not in wanted:
            continue
        selected.append(SelectedFold(fold_id=fold_id, year=int(fold_id.rsplit("_", 1)[-1])))
    if wanted:
        found = {fold.fold_id for fold in selected}
        missing = wanted - found
        if missing:
            raise ValueError(f"unknown fold ids: {sorted(missing)}")
    return selected


def _matrix_manifest_base(source_manifest: dict[str, object]) -> dict[str, object]:
    denied = {
        "artifacts",
        "model_mode",
        "prediction_rows",
        "models_trained_at",
        "predicted_at",
        "backtested_at",
        "metrics_written_at",
    }
    return {key: value for key, value in source_manifest.items() if key not in denied}


def _rolling_years_for_mode(train_window_mode: str) -> int:
    if train_window_mode == "rolling_5y_no_gap":
        return 5
    if train_window_mode == "rolling_3y_no_gap":
        return 3
    raise ValueError(f"unknown rolling train_window_mode: {train_window_mode}")


def _artifact_manifest(artifacts: AlphaRiskFoldArtifacts) -> dict[str, dict[str, object]]:
    return {
        "absolute": _artifact_payload(artifacts.absolute),
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


def _artifact_from_payload(payload: dict[str, object]) -> ModelArtifact:
    artifact_uri = Path(str(payload["artifact_uri"]))
    if not artifact_uri.exists():
        raise FileNotFoundError(artifact_uri)
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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
