from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ml_stock_selector.models.alpha_ranker import load_alpha_ranker
from ml_stock_selector.models.artifacts import ModelArtifact
from ml_stock_selector.storage import upsert_dataframe


def predict_with_model(feature_mart: pd.DataFrame, artifact: ModelArtifact) -> pd.Series:
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


def upsert_predictions(con, rows: pd.DataFrame) -> None:
    upsert_dataframe(con, "ml_predictions_daily", rows, ["trade_date", "code", "model_id", "horizon_d"])

