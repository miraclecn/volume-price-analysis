from __future__ import annotations

from datetime import datetime, timezone
import pickle

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse

from ml_stock_selector.constants import MODEL_TYPE_RISK, SCORE_VERSION_THREE_MODEL
from ml_stock_selector.feature_matrix import load_feature_schema
from ml_stock_selector.matrix_cache import FoldMatrixCache, load_test_matrix
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


def write_chunked_fold_predictions(
    con,
    cache: FoldMatrixCache,
    absolute_artifact,
    active_artifact,
    risk_artifact,
    *,
    score_version: str = SCORE_VERSION_THREE_MODEL,
    chunk_size: int = 50000,
) -> int:
    x_test = load_test_matrix(cache)
    metadata = _read_parquet(cache.metadata_test_path)
    schema = load_feature_schema(cache.feature_schema_path)
    abs_model = _load_pickle_model(absolute_artifact.artifact_uri)
    active_model = _load_pickle_model(active_artifact.artifact_uri)
    risk_model = _load_pickle_model(risk_artifact.artifact_uri)
    rows_written = 0
    for start in range(0, x_test.shape[0], chunk_size):
        end = min(start + chunk_size, x_test.shape[0])
        matrix = _matrix_chunk_to_frame(x_test[start:end], schema.output_columns)
        meta = metadata.iloc[start:end].reset_index(drop=True)
        now = datetime.now(timezone.utc).isoformat()
        raw = meta[["trade_date", "code"]].copy()
        raw["run_id"] = cache.run_id
        raw["fold_id"] = cache.fold_id
        raw["score_version"] = score_version
        raw["feature_set_id"] = absolute_artifact.feature_set_id
        raw["horizon_d"] = absolute_artifact.horizon_d
        raw["absolute_model_id"] = absolute_artifact.model_id
        raw["active_model_id"] = active_artifact.model_id
        raw["risk_model_id"] = risk_artifact.model_id
        raw["absolute_score"] = list(abs_model.predict_matrix(matrix))
        raw["active_score"] = list(active_model.predict_matrix(matrix))
        raw["risk_prob"] = list(risk_model.predict_proba_matrix(matrix))
        raw["generated_at"] = now
        upsert_dataframe(con, "ml_prediction_raw_daily", raw, ["trade_date", "code", "run_id", "fold_id", "horizon_d"])
        rows_written += len(raw)
    rank_raw_predictions_sql(con, cache.run_id, cache.fold_id, score_version)
    return rows_written


def rank_raw_predictions_sql(con, run_id: str, fold_id: str, score_version: str = SCORE_VERSION_THREE_MODEL) -> None:
    con.execute(
        """
        delete from ml_predictions_daily
        where run_id = ? and fold_id = ? and score_version = ?
        """,
        [run_id, fold_id, score_version],
    )
    con.execute(
        """
        delete from ml_predictions_daily
        where (model_id, horizon_d) in (
            select distinct
                'three_model:' || absolute_model_id || ':' || active_model_id || ':' || risk_model_id as model_id,
                horizon_d
            from ml_prediction_raw_daily
            where run_id = ? and fold_id = ? and score_version = ?
        )
        """,
        [run_id, fold_id, score_version],
    )
    con.execute(
        """
        insert into ml_predictions_daily (
            trade_date, code, model_id, horizon_d,
            alpha_score, alpha_rank_pct,
            absolute_score, absolute_rank_pct,
            reg_score,
            active_score, active_rank_pct,
            risk_score, risk_prob, risk_rank_pct,
            context_score, liquidity_score, relative_strength_pct, resonance_pct, penalty_score,
            core_score, trade_score, trade_score_v2,
            score_version, run_id, fold_id,
            absolute_model_id, active_model_id, risk_model_id,
            feature_set_id, generated_at
        )
        select
            trade_date,
            code,
            'three_model:' || absolute_model_id || ':' || active_model_id || ':' || risk_model_id as model_id,
            horizon_d,
            absolute_score as alpha_score,
            absolute_rank_pct as alpha_rank_pct,
            absolute_score,
            absolute_rank_pct,
            null as reg_score,
            active_score,
            active_rank_pct,
            risk_prob as risk_score,
            risk_prob,
            risk_rank_pct,
            null as context_score,
            null as liquidity_score,
            0.5 as relative_strength_pct,
            0.5 as resonance_pct,
            0.0 as penalty_score,
            0.55 * absolute_rank_pct + 0.35 * active_rank_pct - 0.25 * risk_rank_pct as core_score,
            null as trade_score,
            0.55 * absolute_rank_pct + 0.35 * active_rank_pct - 0.25 * risk_rank_pct as trade_score_v2,
            score_version,
            run_id,
            fold_id,
            absolute_model_id,
            active_model_id,
            risk_model_id,
            feature_set_id,
            generated_at
        from (
            select
                *,
                percent_rank() over (partition by trade_date order by absolute_score) as absolute_rank_pct,
                percent_rank() over (partition by trade_date order by active_score) as active_rank_pct,
                percent_rank() over (partition by trade_date order by risk_prob) as risk_rank_pct
            from ml_prediction_raw_daily
            where run_id = ? and fold_id = ? and score_version = ?
        ) ranked
        """,
        [run_id, fold_id, score_version],
    )


def _load_pickle_model(path) -> object:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _matrix_chunk_to_frame(matrix, columns: list[str]) -> pd.DataFrame:
    if sparse.issparse(matrix):
        values = matrix.toarray().astype(np.float32, copy=False)
    else:
        values = np.asarray(matrix, dtype=np.float32)
    return pd.DataFrame(values, columns=columns)


def _read_parquet(path) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        return con.execute("select * from read_parquet(?) order by trade_date, code", [str(path)]).fetchdf()
    finally:
        con.close()


def _cross_sectional_zscore(frame: pd.DataFrame, score_column: str) -> pd.Series:
    grouped = frame.groupby("trade_date")[score_column]
    mean = grouped.transform("mean")
    std = grouped.transform(lambda values: values.std(ddof=0))
    zscore = (frame[score_column] - mean) / std.replace(0.0, pd.NA)
    return zscore.fillna(0.0).clip(-3.0, 3.0)
