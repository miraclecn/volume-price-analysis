from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ml_stock_selector.constants import MODEL_TYPE_RISK, SCORE_VERSION_THREE_MODEL
from ml_stock_selector.models.alpha_ranker import load_alpha_ranker
from ml_stock_selector.models.artifacts import ModelArtifact
from ml_stock_selector.models.calibrator import cross_sectional_percentile
from ml_stock_selector.models.risk_model import load_risk_model
from ml_stock_selector.storage import upsert_dataframe


def predict_with_model(feature_mart: pd.DataFrame, artifact: ModelArtifact) -> pd.Series:
    if artifact.model_type == MODEL_TYPE_RISK:
        model = load_risk_model(artifact)
        return model.predict_proba(feature_mart)
    model = load_alpha_ranker(artifact)
    return model.predict(feature_mart)


def build_prediction_rows(feature_mart: pd.DataFrame, scores: pd.Series, artifact: ModelArtifact) -> pd.DataFrame:
    rows = feature_mart[["trade_date", "code", "feature_set_id"]].copy()
    rows["model_id"] = artifact.model_id
    rows["horizon_d"] = artifact.horizon_d
    rows["alpha_score"] = list(scores)
    rows["alpha_rank_pct"] = rows.groupby("trade_date")["alpha_score"].rank(pct=True)
    rows["reg_score"] = None
    rows["risk_score"] = 0.0
    rows["risk_rank_pct"] = 0.5
    rows["context_score"] = None
    rows["liquidity_score"] = None
    rows["relative_strength_pct"] = 0.5
    rows["resonance_pct"] = 0.5
    rows["penalty_score"] = 0.0
    rows["trade_score"] = None
    rows["generated_at"] = datetime.now(timezone.utc).isoformat()
    return rows


def build_three_model_prediction_rows(
    feature_mart: pd.DataFrame,
    absolute_scores: pd.Series,
    active_scores: pd.Series,
    risk_probs: pd.Series,
    absolute_artifact: ModelArtifact,
    active_artifact: ModelArtifact,
    risk_artifact: ModelArtifact,
) -> pd.DataFrame:
    rows = feature_mart[["trade_date", "code", "feature_set_id"]].copy()
    rows["model_id"] = f"three_model:{absolute_artifact.model_id}:{active_artifact.model_id}:{risk_artifact.model_id}"
    rows["horizon_d"] = absolute_artifact.horizon_d
    rows["absolute_score"] = list(absolute_scores)
    rows["absolute_rank_pct"] = cross_sectional_percentile(rows, "absolute_score")
    rows["absolute_zscore"] = _cross_sectional_zscore(rows, "absolute_score")
    rows["active_score"] = list(active_scores)
    rows["active_rank_pct"] = cross_sectional_percentile(rows, "active_score")
    rows["active_zscore"] = _cross_sectional_zscore(rows, "active_score")
    rows["risk_prob"] = list(risk_probs)
    rows["risk_score"] = rows["risk_prob"]
    rows["risk_rank_pct"] = cross_sectional_percentile(rows, "risk_prob")
    rows["risk_zscore"] = _cross_sectional_zscore(rows, "risk_prob")
    rows["alpha_score"] = rows["absolute_score"]
    rows["alpha_rank_pct"] = rows["absolute_rank_pct"]
    rows["reg_score"] = None
    rows["context_score"] = None
    rows["liquidity_score"] = None
    rows["relative_strength_pct"] = 0.5
    rows["resonance_pct"] = 0.5
    rows["penalty_score"] = 0.0
    rows["trade_score"] = None
    rows["core_score"] = None
    rows["trade_score_v2"] = None
    rows["score_version"] = SCORE_VERSION_THREE_MODEL
    rows["generated_at"] = datetime.now(timezone.utc).isoformat()
    return rows


def upsert_predictions(con, rows: pd.DataFrame) -> None:
    upsert_dataframe(con, "ml_predictions_daily", rows, ["trade_date", "code", "model_id", "horizon_d"])


def _cross_sectional_zscore(frame: pd.DataFrame, score_column: str) -> pd.Series:
    grouped = frame.groupby("trade_date")[score_column]
    mean = grouped.transform("mean")
    std = grouped.transform(lambda values: values.std(ddof=0))
    zscore = (frame[score_column] - mean) / std.replace(0.0, pd.NA)
    return zscore.fillna(0.0).clip(-3.0, 3.0)
