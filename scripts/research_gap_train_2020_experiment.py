from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_stock_selector.config import load_ml_config
from ml_stock_selector.matrix_cache import FoldMatrixCache, load_cached_matrix, update_fold_manifest_status
from ml_stock_selector.models.artifacts import ModelArtifact
from ml_stock_selector.models.config import LightGBMRankerConfig, LightGBMRiskConfig
from ml_stock_selector.models.fold_cache_training import (
    AlphaRiskFoldArtifacts,
    _save_ranker_artifact,
    _save_risk_artifact,
)
from ml_stock_selector.prediction import write_chunked_alpha_risk_fold_predictions
from ml_stock_selector.storage import init_ml_db
from ml_stock_selector.constants import MODEL_TYPE_RANKER

from scripts import research_continuous_profit_protect_backtest as continuous


SOURCE_RUN_ID = "wf_v2_ret5_fund_fixed_a160_r120_20260621"
SOURCE_SCORE_VERSION = "v2_alpha_ret5d_fund_fixed_a160_r120_20260621"
SOURCE_CACHE_ROOT = Path("outputs/ml/cache/folds_ret5_fundamental_fixed_rounds_20260621")
DEFAULT_ML_DB = Path("outputs/ml/ml_ret5_alpha_risk_20260619.duckdb")
DEFAULT_CONFIG = "config/ml_walkforward_adv10m_ret5_alpha_risk.toml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-db", default=str(DEFAULT_ML_DB))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=200_000)
    parser.add_argument("--prediction-chunk-size", type=int, default=50_000)
    parser.add_argument("--source-cache-root", default=str(SOURCE_CACHE_ROOT))
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--target-cache-root", default="outputs/ml/cache/gap_train_walkforward_experiment_20260623")
    parser.add_argument("--artifact-root", default="outputs/ml/artifacts/gap_train_walkforward_experiment_20260623")
    parser.add_argument("--report-root", default="outputs/ml/reports/gap_train_walkforward_experiment_20260623")
    args = parser.parse_args()

    cases = [
        ("one_year_gap", lambda year: f"{year - 2:04d}-12-31"),
        ("half_year_gap", lambda year: f"{year - 1:04d}-06-30"),
    ]
    summaries = []
    years = list(range(int(args.start_year), int(args.end_year) + 1))
    for case_name, train_end_for_year in cases:
        run_id = f"wf_v2_ret5_fund_fixed_a160_r120_{case_name}_20260623"
        score_version = f"v2_alpha_ret5d_fund_fixed_a160_r120_{case_name}_20260623"
        out_dir = Path(args.report_root) / case_name
        train_ends = {}
        for year in years:
            train_end = train_end_for_year(year)
            train_ends[str(year)] = train_end
            run_fold_case(
                case_name=case_name,
                year=year,
                train_end=train_end,
                run_id=run_id,
                score_version=score_version,
                args=args,
            )
        run_continuous_case(
            case_name=case_name,
            run_id=run_id,
            score_version=score_version,
            args=args,
            out_dir=out_dir,
        )
        summary_path = out_dir / "continuous_summary.csv"
        yearly_path = out_dir / "continuous_yearly.csv"
        summary = pd.read_csv(summary_path)
        yearly = pd.read_csv(yearly_path)
        row = summary.iloc[0].to_dict()
        row.update(
            {
                "case": case_name,
                "train_ends": json.dumps(train_ends, sort_keys=True),
                "run_id": run_id,
                "score_version": score_version,
            }
        )
        summaries.append(row)

    root = Path(args.report_root)
    root.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(summaries)
    comparison.to_csv(root / "gap_train_walkforward_comparison.csv", index=False)
    write_baseline_comparison(root, comparison)
    print(comparison.to_string(index=False))
    print(f"wrote {root}")


def run_fold_case(
    *,
    case_name: str,
    year: int,
    train_end: str,
    run_id: str,
    score_version: str,
    args: argparse.Namespace,
) -> FoldMatrixCache:
    fold_id = f"wf_{year}"
    source = FoldMatrixCache.from_paths(SOURCE_RUN_ID, fold_id, Path(args.source_cache_root))
    target = FoldMatrixCache.from_paths(run_id, fold_id, Path(args.target_cache_root))
    if args.force and target.cache_dir.exists():
        shutil.rmtree(target.cache_dir)
    if not target.manifest_path.exists():
        build_filtered_train_cache(source, target, train_end=train_end, chunk_rows=int(args.chunk_rows))

    artifact_dir = Path(args.artifact_root) / case_name / fold_id
    if args.force and artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    manifest = json.loads(target.manifest_path.read_text(encoding="utf-8"))
    if args.force or "artifacts" not in manifest:
        artifacts = train_fixed_round_models(target, load_ml_config(args.config), artifact_dir)
        update_fold_manifest_status(
            target.manifest_path,
            "models_trained",
            artifacts=artifact_manifest(artifacts),
            model_mode="alpha_risk_fixed_rounds_gap_experiment",
            fixed_alpha_rounds=160,
            fixed_risk_rounds=120,
        )
    else:
        artifacts = load_artifacts_from_manifest(manifest)

    con = init_ml_db(args.ml_db)
    try:
        if args.force:
            con.execute("delete from ml_prediction_raw_daily where run_id = ? and fold_id = ? and score_version = ?", [run_id, fold_id, score_version])
            con.execute("delete from ml_predictions_daily where run_id = ? and fold_id = ? and score_version = ?", [run_id, fold_id, score_version])
        existing = con.execute(
            "select count(*) from ml_predictions_daily where run_id = ? and fold_id = ? and score_version = ?",
            [run_id, fold_id, score_version],
        ).fetchone()[0]
        if not existing:
            rows = write_chunked_alpha_risk_fold_predictions(
                con,
                target,
                artifacts.absolute,
                artifacts.risk,
                score_version=score_version,
                chunk_size=int(args.prediction_chunk_size),
            )
            update_fold_manifest_status(target.manifest_path, "predicted", prediction_rows=int(rows), score_version=score_version)
    finally:
        con.close()
    return target


def run_continuous_case(
    *,
    case_name: str,
    run_id: str,
    score_version: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    continuous.main_with_args(
        [
            "--ml-db",
            str(args.ml_db),
            "--run-id",
            run_id,
            "--score-version",
            score_version,
            "--base-portfolio-id",
            f"gap_train_wf_{case_name}",
            "--variant",
            "mkt_tier_profit_protect",
            "--start-year",
            str(args.start_year),
            "--end-year",
            str(args.end_year),
            "--initial-cash",
            "1000000",
            "--slippage-bps",
            "10",
            "--commission-bps",
            "3",
            "--stamp-duty-bps",
            "5",
            "--out-dir",
            str(out_dir),
        ]
    )


def write_baseline_comparison(root: Path, comparison: pd.DataFrame) -> None:
    baseline_dir = Path("outputs/ml/reports/profit_protect_continuous_wf_2020_2025_20260622")
    baseline_summary_path = baseline_dir / "continuous_summary.csv"
    baseline_yearly_path = baseline_dir / "continuous_yearly.csv"
    rows = []
    if baseline_summary_path.exists():
        row = pd.read_csv(baseline_summary_path).iloc[0].to_dict()
        row.update(
            {
                "case": "baseline_full_no_gap",
                "train_ends": "train through prior year end per fold",
                "run_id": SOURCE_RUN_ID,
                "score_version": SOURCE_SCORE_VERSION,
            }
        )
        rows.append(row)
    rows.extend(comparison.to_dict("records"))
    pd.DataFrame(rows).to_csv(root / "gap_train_walkforward_with_baseline.csv", index=False)
    if baseline_yearly_path.exists():
        yearly_frames = [pd.read_csv(baseline_yearly_path).assign(case="baseline_full_no_gap")]
        for case_name in comparison["case"].astype(str):
            path = root / case_name / "continuous_yearly.csv"
            if path.exists():
                yearly_frames.append(pd.read_csv(path).assign(case=case_name))
        pd.concat(yearly_frames, ignore_index=True).to_csv(root / "gap_train_walkforward_yearly_with_baseline.csv", index=False)


def build_filtered_train_cache(source: FoldMatrixCache, target: FoldMatrixCache, *, train_end: str, chunk_rows: int) -> None:
    target.cache_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(source.manifest_path.read_text(encoding="utf-8"))
    for path in [
        source.x_valid_dense_path,
        source.x_test_dense_path,
        source.y_abs_valid_path,
        source.y_active_valid_path,
        source.y_risk_valid_path,
        source.group_valid_path,
        source.metadata_valid_path,
        source.metadata_test_path,
        source.feature_schema_path,
    ]:
        _copy_or_link(path, target.cache_dir / path.name)
    if (source.cache_dir / "y_alpha_eval_valid_future_max_gain.npy").exists():
        _copy_or_link(source.cache_dir / "y_alpha_eval_valid_future_max_gain.npy", target.cache_dir / "y_alpha_eval_valid_future_max_gain.npy")

    matrix = load_cached_matrix(source.x_train_dense_path, source.x_train_path)
    metadata = pd.read_parquet(source.metadata_train_path)
    dates = metadata["trade_date"].astype(str)
    train_start = str(source_manifest["train_start"])
    mask = ((dates >= train_start) & (dates <= train_end)).to_numpy()
    total_rows = int(mask.sum())
    if total_rows <= 0:
        raise ValueError(f"empty filtered train window: {train_start}..{train_end}")

    tmp_path = target.x_train_dense_path.with_name(f".{target.x_train_dense_path.stem}.{uuid4().hex}.npy")
    out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(total_rows, int(matrix.shape[1])))
    offset = 0
    try:
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
            print(f"matrix {target.run_id} rows={offset}/{total_rows}", flush=True)
        del out
        os.replace(tmp_path, target.x_train_dense_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    metadata_out = metadata.loc[mask].copy()
    metadata_out.to_parquet(target.metadata_train_path, index=False)
    for name in ["y_abs_train.npy", "y_active_train.npy", "y_risk_train.npy"]:
        values = np.load(source.cache_dir / name, mmap_mode="r")[mask].astype(np.float32, copy=False)
        np.save(target.cache_dir / name, values)
    np.save(target.group_train_path, metadata_out.groupby("trade_date", sort=False).size().to_numpy(dtype=np.int32))

    payload = {
        **_matrix_only_manifest(source_manifest),
        "run_id": target.run_id,
        "fold_id": target.fold_id,
        "status": "matrix_built",
        "source_run_id": source.run_id,
        "source_cache_dir": str(source.cache_dir),
        "train_window_mode": "custom_gap_before_test",
        "train_start": train_start,
        "train_end": train_end,
        "train_rows": total_rows,
        "fixed_round_note": "Gap experiment: train matrix is filtered from existing fixed-round full-history cache; validation is not used for early stopping.",
        "matrix_built_at": datetime.now(timezone.utc).isoformat(),
    }
    target.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _matrix_only_manifest(source_manifest: dict[str, object]) -> dict[str, object]:
    payload = dict(source_manifest)
    for key in [
        "artifacts",
        "models_trained_at",
        "predicted_at",
        "prediction_rows",
        "score_version",
        "backtested_at",
        "metrics_written_at",
    ]:
        payload.pop(key, None)
    return payload


def train_fixed_round_models(cache: FoldMatrixCache, config, artifact_dir: Path) -> AlphaRiskFoldArtifacts:
    model = dict(config.model)
    runtime = dict(model.get("lightgbm_runtime", {}))
    runtime.update({"early_stopping_rounds": 0})
    ranker_raw = {**dict(model.get("alpha_ranker", {})), **runtime, "n_estimators": 160}
    risk_raw = {**dict(model.get("risk_model", {})), **runtime, "n_estimators": 120}
    ranker_config = LightGBMRankerConfig(
        objective=str(ranker_raw.get("objective", "lambdarank")),
        metric=str(ranker_raw.get("metric", "ndcg")),
        n_estimators=int(ranker_raw["n_estimators"]),
        learning_rate=float(ranker_raw.get("learning_rate", 0.05)),
        num_leaves=int(ranker_raw.get("num_leaves", 31)),
        min_data_in_leaf=int(ranker_raw.get("min_data_in_leaf", 500)),
        feature_fraction=float(ranker_raw["feature_fraction"]) if "feature_fraction" in ranker_raw else None,
        bagging_fraction=float(ranker_raw["bagging_fraction"]) if "bagging_fraction" in ranker_raw else None,
        bagging_freq=int(ranker_raw["bagging_freq"]) if "bagging_freq" in ranker_raw else None,
        lambda_l2=float(ranker_raw["lambda_l2"]) if "lambda_l2" in ranker_raw else None,
        eval_at=tuple(int(x) for x in ranker_raw.get("eval_at", [10, 15])),
        lambdarank_truncation_level=int(ranker_raw["lambdarank_truncation_level"]) if "lambdarank_truncation_level" in ranker_raw else None,
        early_stopping_rounds=0,
        num_threads=int(ranker_raw["num_threads"]) if "num_threads" in ranker_raw else None,
        force_col_wise=bool(ranker_raw["force_col_wise"]) if "force_col_wise" in ranker_raw else None,
        max_bin=int(ranker_raw["max_bin"]) if "max_bin" in ranker_raw else None,
        histogram_pool_size=int(ranker_raw["histogram_pool_size"]) if "histogram_pool_size" in ranker_raw else None,
    )
    risk_config = LightGBMRiskConfig(
        objective=str(risk_raw.get("objective", "binary")),
        n_estimators=int(risk_raw["n_estimators"]),
        learning_rate=float(risk_raw.get("learning_rate", 0.05)),
        num_leaves=int(risk_raw.get("num_leaves", 31)),
        min_data_in_leaf=int(risk_raw.get("min_data_in_leaf", 500)),
        feature_fraction=float(risk_raw["feature_fraction"]) if "feature_fraction" in risk_raw else None,
        bagging_fraction=float(risk_raw["bagging_fraction"]) if "bagging_fraction" in risk_raw else None,
        bagging_freq=int(risk_raw["bagging_freq"]) if "bagging_freq" in risk_raw else None,
        lambda_l2=float(risk_raw["lambda_l2"]) if "lambda_l2" in risk_raw else None,
        early_stopping_rounds=0,
        num_threads=int(risk_raw["num_threads"]) if "num_threads" in risk_raw else None,
        force_col_wise=bool(risk_raw["force_col_wise"]) if "force_col_wise" in risk_raw else None,
        max_bin=int(risk_raw["max_bin"]) if "max_bin" in risk_raw else None,
        histogram_pool_size=int(risk_raw["histogram_pool_size"]) if "histogram_pool_size" in risk_raw else None,
    )
    x_train = load_cached_matrix(cache.x_train_dense_path, cache.x_train_path)
    x_valid = load_cached_matrix(cache.x_valid_dense_path, cache.x_valid_path)
    y_abs = np.load(cache.y_abs_train_path)
    y_risk = np.load(cache.y_risk_train_path)
    y_abs_valid = np.load(cache.y_abs_valid_path)
    y_risk_valid = np.load(cache.y_risk_valid_path)
    group_train = np.load(cache.group_train_path)
    group_valid = np.load(cache.group_valid_path)
    schema_payload = json.loads(cache.feature_schema_path.read_text(encoding="utf-8"))
    feature_columns = list(schema_payload["output_columns"])
    manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    absolute = _save_ranker_artifact(
        "alpha_ranker",
        MODEL_TYPE_RANKER,
        x_train,
        y_abs,
        group_train,
        x_valid,
        y_abs_valid,
        group_valid,
        feature_columns,
        str(manifest["feature_set_id"]),
        "absolute_label",
        str(manifest["label_base"]),
        int(manifest["horizon_d"]),
        cache.feature_schema_path,
        artifact_dir,
        ranker_config,
    )
    risk = _save_risk_artifact(
        x_train,
        y_risk,
        x_valid,
        y_risk_valid,
        feature_columns,
        str(manifest["feature_set_id"]),
        "risk_label",
        str(manifest["label_base"]),
        int(manifest["horizon_d"]),
        cache.feature_schema_path,
        artifact_dir,
        risk_config,
    )
    return AlphaRiskFoldArtifacts(absolute=absolute, risk=risk)


def artifact_manifest(artifacts: AlphaRiskFoldArtifacts) -> dict[str, dict[str, object]]:
    def payload(artifact: ModelArtifact) -> dict[str, object]:
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

    return {"absolute": payload(artifacts.absolute), "risk": payload(artifacts.risk)}


def load_artifacts_from_manifest(manifest: dict[str, object]) -> AlphaRiskFoldArtifacts:
    raw = manifest["artifacts"]

    def artifact(payload: dict[str, object]) -> ModelArtifact:
        return ModelArtifact(
            str(payload["model_id"]),
            str(payload["model_type"]),
            str(payload["feature_set_id"]),
            str(payload["label_name"]),
            str(payload["label_base"]),
            int(payload["horizon_d"]),
            Path(str(payload["feature_schema_uri"])),
            Path(str(payload["artifact_uri"])),
            Path(str(payload["artifact_dir"])),
            dict(payload.get("metrics", {})),
        )

    return AlphaRiskFoldArtifacts(absolute=artifact(raw["absolute"]), risk=artifact(raw["risk"]))


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
