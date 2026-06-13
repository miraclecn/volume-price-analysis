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
    ranker = LGBMRanker(
        verbose=-1,
        **params,
    )
    try:
        fit_kwargs = {"group": group_train.tolist(), "eval_at": eval_at}
        if early_stopping_rounds > 0 and x_valid.shape[0] > 0 and group_valid.size > 0:
            from lightgbm import early_stopping

            fit_kwargs.update(
                {
                    "eval_set": [(x_valid, valid_target.astype(int))],
                    "eval_group": [group_valid.tolist()],
                    "callbacks": [early_stopping(early_stopping_rounds, verbose=False)],
                }
            )
        ranker.fit(x_train, target.astype(int), **fit_kwargs)
    except Exception:
        return None
    return LightGBMRankerAdapter(ranker, feature_columns)


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
    return metrics


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
