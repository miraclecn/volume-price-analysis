from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
import json
import pickle

import numpy as np
from scipy import sparse

from ml_stock_selector.constants import MODEL_TYPE_ACTIVE_RANKER, MODEL_TYPE_RANKER, MODEL_TYPE_RISK
from ml_stock_selector.matrix_cache import FoldMatrixCache, load_train_matrix, load_valid_matrix
from ml_stock_selector.models.alpha_ranker import LightGBMRankerAdapter, LinearFallbackModel
from ml_stock_selector.models.artifacts import ModelArtifact
from ml_stock_selector.models.config import (
    LightGBMRankerConfig,
    LightGBMRiskConfig,
    ranker_config_from_model_section,
    risk_config_from_model_section,
)
from ml_stock_selector.models.risk_model import LightGBMClassifierAdapter, LogisticFallbackModel


@dataclass(frozen=True)
class ThreeModelFoldArtifacts:
    absolute: ModelArtifact
    active: ModelArtifact
    risk: ModelArtifact

    @property
    def model_ids(self) -> list[str]:
        return [self.absolute.model_id, self.active.model_id, self.risk.model_id]


@dataclass(frozen=True)
class AlphaRiskFoldArtifacts:
    absolute: ModelArtifact
    risk: ModelArtifact

    @property
    def model_ids(self) -> list[str]:
        return [self.absolute.model_id, self.risk.model_id]


def train_three_models_from_fold_cache(
    cache: FoldMatrixCache,
    config,
    artifact_dir: str | Path,
) -> ThreeModelFoldArtifacts:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    x_train = load_train_matrix(cache)
    y_abs = np.load(cache.y_abs_train_path)
    y_active = np.load(cache.y_active_train_path)
    y_risk = np.load(cache.y_risk_train_path)
    group_train = np.load(cache.group_train_path)
    x_valid = load_valid_matrix(cache)
    y_abs_valid = np.load(cache.y_abs_valid_path)
    y_active_valid = np.load(cache.y_active_valid_path)
    y_risk_valid = np.load(cache.y_risk_valid_path)
    group_valid = np.load(cache.group_valid_path)
    schema_payload = json.loads(cache.feature_schema_path.read_text(encoding="utf-8"))
    feature_columns = list(schema_payload["output_columns"])
    manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    alpha_eval_target = _load_optional_manifest_array(cache, manifest.get("alpha_eval_valid_path"))
    alpha_eval_metric = str(manifest.get("alpha_eval_metric") or "")
    feature_set_id = str(manifest["feature_set_id"])
    label_base = str(manifest["label_base"])
    horizon_d = int(manifest["horizon_d"])
    ranker_config = ranker_config_from_model_section(getattr(config, "model", {}))
    risk_config = risk_config_from_model_section(getattr(config, "model", {}))

    return ThreeModelFoldArtifacts(
        absolute=_save_ranker_artifact(
            "alpha_ranker",
            MODEL_TYPE_RANKER,
            x_train,
            y_abs,
            group_train,
            x_valid,
            y_abs_valid,
            group_valid,
            feature_columns,
            feature_set_id,
            "absolute_label",
            label_base,
            horizon_d,
            cache.feature_schema_path,
            artifact_dir,
            ranker_config,
            valid_eval_target=alpha_eval_target,
            eval_metric_name=alpha_eval_metric,
        ),
        active=_save_ranker_artifact(
            "active_ranker",
            MODEL_TYPE_ACTIVE_RANKER,
            x_train,
            y_active,
            group_train,
            x_valid,
            y_active_valid,
            group_valid,
            feature_columns,
            feature_set_id,
            "active_label",
            label_base,
            horizon_d,
            cache.feature_schema_path,
            artifact_dir,
            ranker_config,
        ),
        risk=_save_risk_artifact(
            x_train,
            y_risk,
            x_valid,
            y_risk_valid,
            feature_columns,
            feature_set_id,
            "risk_label",
            label_base,
            horizon_d,
            cache.feature_schema_path,
            artifact_dir,
            risk_config,
        ),
    )


def train_alpha_risk_models_from_fold_cache(
    cache: FoldMatrixCache,
    config,
    artifact_dir: str | Path,
) -> AlphaRiskFoldArtifacts:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    x_train = load_train_matrix(cache)
    y_abs = np.load(cache.y_abs_train_path)
    y_risk = np.load(cache.y_risk_train_path)
    group_train = np.load(cache.group_train_path)
    x_valid = load_valid_matrix(cache)
    y_abs_valid = np.load(cache.y_abs_valid_path)
    y_risk_valid = np.load(cache.y_risk_valid_path)
    group_valid = np.load(cache.group_valid_path)
    schema_payload = json.loads(cache.feature_schema_path.read_text(encoding="utf-8"))
    feature_columns = list(schema_payload["output_columns"])
    manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    alpha_eval_target = _load_optional_manifest_array(cache, manifest.get("alpha_eval_valid_path"))
    alpha_eval_metric = str(manifest.get("alpha_eval_metric") or "")
    if alpha_eval_metric and alpha_eval_target is None:
        raise FileNotFoundError(f"alpha eval target is missing for {cache.fold_id}: {manifest.get('alpha_eval_valid_path')}")
    if alpha_eval_target is not None and len(alpha_eval_target) != len(y_abs_valid):
        raise ValueError(
            f"alpha eval target row count mismatch for {cache.fold_id}: "
            f"eval={len(alpha_eval_target)} valid={len(y_abs_valid)}"
        )
    feature_set_id = str(manifest["feature_set_id"])
    label_base = str(manifest["label_base"])
    horizon_d = int(manifest["horizon_d"])
    ranker_config = ranker_config_from_model_section(getattr(config, "model", {}))
    risk_config = risk_config_from_model_section(getattr(config, "model", {}))

    return AlphaRiskFoldArtifacts(
        absolute=_save_ranker_artifact(
            "alpha_ranker",
            MODEL_TYPE_RANKER,
            x_train,
            y_abs,
            group_train,
            x_valid,
            y_abs_valid,
            group_valid,
            feature_columns,
            feature_set_id,
            "absolute_label",
            label_base,
            horizon_d,
            cache.feature_schema_path,
            artifact_dir,
            ranker_config,
            valid_eval_target=alpha_eval_target,
            eval_metric_name=alpha_eval_metric,
        ),
        risk=_save_risk_artifact(
            x_train,
            y_risk,
            x_valid,
            y_risk_valid,
            feature_columns,
            feature_set_id,
            "risk_label",
            label_base,
            horizon_d,
            cache.feature_schema_path,
            artifact_dir,
            risk_config,
        ),
    )


def _save_ranker_artifact(
    prefix: str,
    model_type: str,
    x_train,
    target: np.ndarray,
    group_train: np.ndarray,
    x_valid,
    valid_target: np.ndarray,
    group_valid: np.ndarray,
    feature_columns: list[str],
    feature_set_id: str,
    label_name: str,
    label_base: str,
    horizon_d: int,
    schema_path: Path,
    artifact_dir: Path,
    train_config: LightGBMRankerConfig,
    valid_eval_target: np.ndarray | None = None,
    eval_metric_name: str = "",
) -> ModelArtifact:
    model_id = f"{prefix}_{uuid4().hex[:12]}"
    model = _train_lightgbm_ranker(
        x_train,
        target,
        group_train,
        x_valid,
        valid_target,
        group_valid,
        feature_columns,
        train_config,
        valid_eval_target,
        eval_metric_name,
    )
    if model is None:
        model = LinearFallbackModel(feature_columns, {col: 0.0 for col in feature_columns}, float(np.nanmean(target) if len(target) else 0.0))
    artifact_path = artifact_dir / f"{model_id}.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(model, handle)
    _write_json(artifact_dir / f"{model_id}.params.json", train_config.to_params())
    return ModelArtifact(
        model_id,
        model_type,
        feature_set_id,
        label_name,
        label_base,
        horizon_d,
        schema_path,
        artifact_path,
        artifact_dir,
        _artifact_metrics(model, x_train.shape[0]),
    )


def _save_risk_artifact(
    x_train,
    target: np.ndarray,
    x_valid,
    valid_target: np.ndarray,
    feature_columns: list[str],
    feature_set_id: str,
    label_name: str,
    label_base: str,
    horizon_d: int,
    schema_path: Path,
    artifact_dir: Path,
    train_config: LightGBMRiskConfig,
) -> ModelArtifact:
    model_id = f"risk_model_{uuid4().hex[:12]}"
    model = _train_lightgbm_classifier(x_train, target, x_valid, valid_target, feature_columns, train_config)
    if model is None:
        model = LogisticFallbackModel(feature_columns, {col: 0.0 for col in feature_columns}, 0.0)
    artifact_path = artifact_dir / f"{model_id}.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(model, handle)
    _write_json(artifact_dir / f"{model_id}.params.json", train_config.to_params())
    return ModelArtifact(
        model_id,
        MODEL_TYPE_RISK,
        feature_set_id,
        label_name,
        label_base,
        horizon_d,
        schema_path,
        artifact_path,
        artifact_dir,
        _artifact_metrics(model, x_train.shape[0]),
    )


def _train_lightgbm_ranker(
    x_train,
    target: np.ndarray,
    group_train: np.ndarray,
    x_valid,
    valid_target: np.ndarray,
    group_valid: np.ndarray,
    feature_columns: list[str],
    train_config: LightGBMRankerConfig,
    valid_eval_target: np.ndarray | None = None,
    eval_metric_name: str = "",
):
    if x_train.shape[0] == 0 or group_train.size == 0:
        return None
    try:
        from lightgbm import LGBMRanker
    except Exception:
        return None
    params = train_config.to_params().copy()
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 0) or 0)
    eval_at = params.pop("eval_at", [10, 15])
    try:
        fit_kwargs = {"group": group_train.tolist(), "eval_at": eval_at}
        if early_stopping_rounds > 0 and x_valid.shape[0] > 0 and group_valid.size > 0:
            from lightgbm import early_stopping

            callbacks = [early_stopping(early_stopping_rounds, verbose=False, first_metric_only=True)]
            if eval_metric_name == "future_max_gain_rank_ic" and valid_eval_target is not None:
                params["metric"] = "None"
                fit_kwargs.update(
                    {
                        "eval_set": [(x_valid, valid_target.astype(int))],
                        "eval_group": [group_valid.tolist()],
                        "eval_metric": _make_group_rank_ic_eval(group_valid, valid_eval_target),
                        "callbacks": callbacks,
                    }
                )
            else:
                fit_kwargs.update(
                    {
                        "eval_set": [(x_valid, valid_target.astype(int))],
                        "eval_group": [group_valid.tolist()],
                        "callbacks": callbacks,
                    }
                )
        ranker = LGBMRanker(
            verbose=-1,
            **params,
        )
        ranker.fit(x_train, target.astype(int), **fit_kwargs)
    except Exception:
        return None
    return LightGBMRankerAdapter(ranker, feature_columns)


def _make_group_rank_ic_eval(group: np.ndarray, eval_target: np.ndarray):
    group_sizes = [int(value) for value in group.tolist()]
    target = np.asarray(eval_target, dtype=np.float32)

    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
        return "future_max_gain_rank_ic", _mean_group_rank_ic(target, y_pred, group_sizes), True

    return evaluate


def _mean_group_rank_ic(y_true: np.ndarray, y_pred: np.ndarray, group_sizes: list[int]) -> float:
    values = []
    offset = 0
    for size in group_sizes:
        end = offset + int(size)
        if size >= 2:
            target = np.asarray(y_true[offset:end], dtype=np.float64)
            pred = np.asarray(y_pred[offset:end], dtype=np.float64)
            if np.isfinite(target).all() and np.isfinite(pred).all():
                target_rank = _average_ranks(target)
                pred_rank = _average_ranks(pred)
                target_std = float(target_rank.std())
                pred_std = float(pred_rank.std())
                if target_std > 0.0 and pred_std > 0.0:
                    corr = float(np.corrcoef(target_rank, pred_rank)[0, 1])
                    if np.isfinite(corr):
                        values.append(corr)
        offset = end
    return float(np.mean(values)) if values else 0.0


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _train_lightgbm_classifier(
    x_train,
    target: np.ndarray,
    x_valid,
    valid_target: np.ndarray,
    feature_columns: list[str],
    train_config: LightGBMRiskConfig,
):
    if x_train.shape[0] == 0 or len(set(target.tolist())) < 2:
        return None
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None
    params = train_config.to_params().copy()
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 0) or 0)
    classifier = LGBMClassifier(
        verbose=-1,
        **params,
    )
    try:
        fit_kwargs = {}
        if early_stopping_rounds > 0 and x_valid.shape[0] > 0 and len(set(valid_target.tolist())) >= 2:
            from lightgbm import early_stopping

            fit_kwargs.update(
                {
                    "eval_set": [(x_valid, valid_target.astype(int))],
                    "eval_metric": "auc",
                    "callbacks": [early_stopping(early_stopping_rounds, verbose=False)],
                }
            )
        classifier.fit(x_train, target.astype(int), **fit_kwargs)
    except Exception:
        return None
    return LightGBMClassifierAdapter(classifier, feature_columns)


def _artifact_metrics(model, train_rows: int) -> dict[str, float]:
    metrics = {"train_rows": float(train_rows)}
    raw_model = getattr(model, "model", None)
    best_iteration = getattr(raw_model, "best_iteration_", None)
    if best_iteration is not None:
        metrics["best_iteration"] = float(best_iteration)
    best_score = getattr(raw_model, "best_score_", None)
    if isinstance(best_score, dict):
        for dataset_name, dataset_scores in best_score.items():
            if isinstance(dataset_scores, dict):
                for metric_name, value in dataset_scores.items():
                    if isinstance(value, (int, float)):
                        metrics[f"best_score_{dataset_name}_{metric_name}"] = float(value)
    return metrics


def _load_optional_manifest_array(cache: FoldMatrixCache, raw_path: object) -> np.ndarray | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = cache.cache_dir / path
    if not path.exists():
        return None
    return np.load(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
