from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
import json
import math
import pickle

import pandas as pd

from ml_stock_selector.constants import MODEL_TYPE_RISK
from ml_stock_selector.feature_matrix import build_feature_matrix, load_feature_schema, save_feature_schema
from ml_stock_selector.models.alpha_ranker import _fit_linear_weights
from ml_stock_selector.models.artifacts import ModelArtifact
from ml_stock_selector.models.config import LightGBMRiskConfig


@dataclass
class LogisticFallbackModel:
    feature_columns: list[str]
    weights: dict[str, float]
    intercept: float = 0.0

    def predict_proba_matrix(self, matrix: pd.DataFrame) -> pd.Series:
        score = pd.Series(self.intercept, index=matrix.index, dtype=float)
        for col in self.feature_columns:
            score = score + matrix[col] * self.weights.get(col, 0.0)
        return score.map(_sigmoid)


@dataclass
class LightGBMClassifierAdapter:
    model: object
    feature_columns: list[str]

    def predict_proba_matrix(self, matrix: pd.DataFrame) -> pd.Series:
        values = self.model.predict_proba(matrix[self.feature_columns])[:, 1]
        return pd.Series(values, index=matrix.index, dtype=float)


class LoadedRiskModel:
    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact
        self.schema = load_feature_schema(artifact.feature_schema_uri)
        with artifact.artifact_uri.open("rb") as handle:
            self.model = pickle.load(handle)

    def predict_proba(self, frame: pd.DataFrame) -> pd.Series:
        matrix, _ = build_feature_matrix(frame, self.artifact.feature_set_id, schema=self.schema, fit=False)
        return self.model.predict_proba_matrix(matrix)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        return self.predict_proba(frame)


def train_risk_model(
    samples: pd.DataFrame,
    feature_set_id: str,
    label_name: str,
    label_base: str,
    horizon_d: int,
    artifact_dir: Path | str,
    deny_industry: bool = False,
    train_config: LightGBMRiskConfig | None = None,
) -> ModelArtifact:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    matrix, schema = build_feature_matrix(samples, feature_set_id, fit=True, deny_industry=deny_industry)
    target = pd.to_numeric(samples[label_name], errors="coerce").fillna(0).astype(int)
    train_config = train_config or LightGBMRiskConfig()
    model = _train_lightgbm_classifier(matrix, target, train_config) or LogisticFallbackModel(
        list(matrix.columns),
        _fit_linear_weights(matrix, target.astype(float)),
        _logit(float(target.mean() if len(target) else 0.0)),
    )
    model_id = f"risk_model_{uuid4().hex[:12]}"
    schema_path = artifact_dir / f"{model_id}_schema.json"
    artifact_path = artifact_dir / f"{model_id}.pkl"
    save_feature_schema(schema, schema_path)
    with artifact_path.open("wb") as handle:
        pickle.dump(model, handle)
    _write_json(artifact_dir / f"{model_id}.params.json", train_config.to_params())
    probs = model.predict_proba_matrix(matrix)
    metrics = _risk_metrics(target, probs)
    return ModelArtifact(model_id, MODEL_TYPE_RISK, feature_set_id, label_name, label_base, horizon_d, schema_path, artifact_path, artifact_dir, metrics)


def load_risk_model(artifact: ModelArtifact) -> LoadedRiskModel:
    return LoadedRiskModel(artifact)


def _train_lightgbm_classifier(matrix: pd.DataFrame, target: pd.Series, train_config: LightGBMRiskConfig):
    if matrix.empty or target.nunique(dropna=True) < 2:
        return None
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None
    params = train_config.to_params().copy()
    params.pop("early_stopping_rounds", None)
    classifier = LGBMClassifier(verbose=-1, **params)
    try:
        classifier.fit(matrix, target)
    except Exception:
        return None
    return LightGBMClassifierAdapter(classifier, list(matrix.columns))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _risk_metrics(target: pd.Series, probs: pd.Series) -> dict[str, float]:
    if target.nunique(dropna=True) < 2:
        return {"roc_auc_unavailable": 1.0}
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return {"roc_auc_unavailable": 1.0}
    return {"roc_auc": float(roc_auc_score(target, probs))}


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(float(value), 50.0), -50.0)))


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))
