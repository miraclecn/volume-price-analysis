from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
import pickle

import pandas as pd

from ml_stock_selector.constants import MODEL_TYPE_RANKER
from ml_stock_selector.feature_matrix import build_feature_matrix, load_feature_schema, save_feature_schema
from ml_stock_selector.models.artifacts import ModelArtifact


@dataclass
class LinearFallbackModel:
    feature_columns: list[str]
    weights: dict[str, float]
    intercept: float = 0.0

    def predict_matrix(self, matrix: pd.DataFrame) -> pd.Series:
        score = pd.Series(self.intercept, index=matrix.index, dtype=float)
        for col in self.feature_columns:
            score = score + matrix[col] * self.weights.get(col, 0.0)
        return score


class LoadedAlphaRanker:
    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact
        self.schema = load_feature_schema(artifact.feature_schema_uri)
        with artifact.artifact_uri.open("rb") as handle:
            self.model: LinearFallbackModel = pickle.load(handle)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        matrix, _ = build_feature_matrix(frame, self.artifact.feature_set_id, schema=self.schema, fit=False)
        return self.model.predict_matrix(matrix)


@dataclass
class LightGBMRankerAdapter:
    model: object
    feature_columns: list[str]

    def predict_matrix(self, matrix: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(matrix[self.feature_columns]), index=matrix.index)


def train_alpha_ranker(
    samples: pd.DataFrame,
    feature_set_id: str,
    label_name: str,
    label_base: str,
    horizon_d: int,
    artifact_dir: Path | str,
    deny_industry: bool = False,
) -> ModelArtifact:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    matrix, schema = build_feature_matrix(samples, feature_set_id, fit=True, deny_industry=deny_industry)
    target = pd.to_numeric(samples[label_name], errors="coerce").fillna(0.0)
    model = _train_lightgbm_ranker(matrix, target, samples) or LinearFallbackModel(
        list(matrix.columns),
        _fit_linear_weights(matrix, target),
        float(target.mean() if len(target) else 0.0),
    )
    model_id = f"alpha_ranker_{uuid4().hex[:12]}"
    schema_path = artifact_dir / f"{model_id}_schema.json"
    artifact_path = artifact_dir / f"{model_id}.pkl"
    save_feature_schema(schema, schema_path)
    with artifact_path.open("wb") as handle:
        pickle.dump(model, handle)
    preds = model.predict_matrix(matrix)
    if len(samples) > 1 and float(preds.std(ddof=0) or 0.0) != 0.0 and float(target.std(ddof=0) or 0.0) != 0.0:
        train_corr = float(preds.corr(target))
    else:
        train_corr = 0.0
    metrics = {
        "train_corr": train_corr,
        "train_rank_ic": _rank_ic(samples, preds, label_name),
        "eval_at_10": 10.0,
        "eval_at_15": 15.0,
    }
    return ModelArtifact(model_id, MODEL_TYPE_RANKER, feature_set_id, label_name, label_base, horizon_d, schema_path, artifact_path, artifact_dir, metrics)


def load_alpha_ranker(artifact: ModelArtifact) -> LoadedAlphaRanker:
    return LoadedAlphaRanker(artifact)


def _fit_linear_weights(matrix: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    weights = {}
    if float(target.std(ddof=0) or 0.0) == 0.0:
        return {col: 0.0 for col in matrix.columns}
    for col in matrix.columns:
        if float(matrix[col].std(ddof=0) or 0.0) == 0.0:
            weights[col] = 0.0
            continue
        corr = matrix[col].corr(target)
        weights[col] = 0.0 if pd.isna(corr) else float(corr)
    return weights


def _train_lightgbm_ranker(matrix: pd.DataFrame, target: pd.Series, samples: pd.DataFrame):
    try:
        from lightgbm import LGBMRanker
    except Exception:
        return None
    if matrix.empty or samples.empty:
        return None
    group = samples.sort_values(["trade_date", "code"]).groupby("trade_date", sort=True).size().tolist()
    train_x = matrix.loc[samples.sort_values(["trade_date", "code"]).index]
    train_y = target.loc[train_x.index]
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=25,
        learning_rate=0.05,
        num_leaves=15,
        min_data_in_leaf=1,
        verbose=-1,
    )
    try:
        ranker.fit(train_x, train_y.astype(int), group=group, eval_at=[10, 15])
    except Exception:
        return None

    return LightGBMRankerAdapter(ranker, list(matrix.columns))


def _rank_ic(samples: pd.DataFrame, preds: pd.Series, label_name: str) -> float:
    target_name = {
        "rank_label": "future_score",
        "absolute_label": "absolute_ret",
        "active_label": "active_score",
    }.get(label_name, label_name)
    if target_name not in samples:
        return 0.0
    frame = samples[["trade_date"]].copy()
    frame["pred"] = list(preds)
    frame["target"] = pd.to_numeric(samples[target_name], errors="coerce")
    values = []
    for _, group in frame.dropna(subset=["pred", "target"]).groupby("trade_date", sort=False):
        if len(group) < 2:
            continue
        pred_rank = group["pred"].rank()
        target_rank = group["target"].rank()
        if float(pred_rank.std(ddof=0) or 0.0) == 0.0 or float(target_rank.std(ddof=0) or 0.0) == 0.0:
            continue
        corr = pred_rank.corr(target_rank)
        if not pd.isna(corr):
            values.append(float(corr))
    return float(sum(values) / len(values)) if values else 0.0
