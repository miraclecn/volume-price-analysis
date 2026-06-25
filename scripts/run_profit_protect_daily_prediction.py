from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_stock_selector.fundamental_features import load_fundamental_features_for_metadata
from ml_stock_selector.models.artifacts import ModelArtifact
from ml_stock_selector.prediction import predict_with_model
from ml_stock_selector.serving.live_sim import (
    PROFIT_PROTECT_PORTFOLIO_ID,
    PROFIT_PROTECT_RUN_ID,
    PROFIT_PROTECT_SCORE_VERSION,
    PROFIT_PROTECT_SERVING_FOLD_ID,
    activate_profit_protect_live_bundle,
    init_live_sim_db,
    upsert_live_predictions,
)
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


DEFAULT_ML_DB = "outputs/ml/ml_ret5_alpha_risk_20260619.duckdb"
DEFAULT_BASE_FEATURE_SET_ID = "vpa_d_sequence"
DEFAULT_FUNDAMENTAL_RAW_DB = "/home/nan/alpha-data-local/output/raw.duckdb"
DEFAULT_ALPHA_MANIFEST = (
    "outputs/ml/cache/folds_ret5_fundamental_fixed_rounds_20260621/"
    "run_id=wf_v2_ret5_fund_fixed_a160_r80_20260621/fold_id=wf_2026/manifest.json"
)
DEFAULT_RISK_MANIFEST = (
    "outputs/ml/cache/folds_ret5_fundamental_fixed_rounds_20260621/"
    "run_id=wf_v2_ret5_fund_fixed_a160_r120_20260621/fold_id=wf_2026/manifest.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-db", default=DEFAULT_ML_DB)
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--base-feature-set-id", default=DEFAULT_BASE_FEATURE_SET_ID)
    parser.add_argument("--fundamental-raw-db", default=DEFAULT_FUNDAMENTAL_RAW_DB)
    parser.add_argument("--alpha-manifest", default=DEFAULT_ALPHA_MANIFEST)
    parser.add_argument("--risk-manifest", default=DEFAULT_RISK_MANIFEST)
    parser.add_argument("--run-id", default=PROFIT_PROTECT_RUN_ID)
    parser.add_argument("--fold-id", default="wf_2026")
    parser.add_argument("--score-version", default=PROFIT_PROTECT_SCORE_VERSION)
    parser.add_argument("--live-db", help="Optional live DB to receive a live-owned recent prediction copy.")
    args = parser.parse_args()

    con = init_ml_db(args.ml_db)
    try:
        feature_mart = load_daily_fundamental_feature_mart(
            con,
            args.as_of_date,
            args.base_feature_set_id,
            args.fundamental_raw_db,
        )
        alpha_artifact = artifact_from_manifest(Path(args.alpha_manifest), "absolute")
        risk_artifact = artifact_from_manifest(Path(args.risk_manifest), "risk")
        predictions = predict_daily_alpha_risk(
            con,
            feature_mart,
            alpha_artifact,
            risk_artifact,
            run_id=args.run_id,
            fold_id=args.fold_id,
            score_version=args.score_version,
        )
    finally:
        con.close()
    live_rows = write_predictions_to_live_db(predictions, args.live_db) if args.live_db else 0
    print(
        " ".join(
            [
                f"date={args.as_of_date}",
                f"rows={len(predictions)}",
                f"live_rows={live_rows}",
                f"run_id={args.run_id}",
                f"fold_id={args.fold_id}",
                f"score_version={args.score_version}",
            ]
        )
    )


def load_daily_fundamental_feature_mart(
    con,
    as_of_date: str,
    base_feature_set_id: str,
    raw_db_path: str,
) -> pd.DataFrame:
    feature_mart = con.execute(
        """
        select *
        from ml_feature_mart_daily
        where trade_date = ?
          and feature_set_id = ?
        order by code
        """,
        [as_of_date, base_feature_set_id],
    ).fetchdf()
    if feature_mart.empty:
        raise RuntimeError(f"no feature rows found for date={as_of_date}, feature_set_id={base_feature_set_id}")

    metadata = feature_mart[["trade_date", "code"]].copy()
    fundamentals = load_fundamental_features_for_metadata(raw_db_path, metadata)
    out = feature_mart.copy()
    out["features_json"] = append_feature_json_columns(out["features_json"], fundamentals)
    out["feature_set_id"] = f"{base_feature_set_id}_fundamental_v1"
    return out


def append_feature_json_columns(features_json: pd.Series, extra_columns: pd.DataFrame) -> pd.Series:
    if len(features_json) != len(extra_columns):
        raise ValueError("features_json and extra_columns length mismatch")
    rows = []
    for raw, (_, extra) in zip(features_json.tolist(), extra_columns.iterrows(), strict=True):
        payload = json.loads(raw or "{}")
        for column, value in extra.to_dict().items():
            payload[column] = 0.0 if pd.isna(value) else float(value)
        rows.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return pd.Series(rows, index=features_json.index)


def artifact_from_manifest(manifest_path: Path, role: str) -> ModelArtifact:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = manifest["artifacts"][role]
    artifact_uri = Path(str(payload["artifact_uri"]))
    feature_schema_uri = Path(str(payload["feature_schema_uri"]))
    if not artifact_uri.exists():
        raise FileNotFoundError(artifact_uri)
    if not feature_schema_uri.exists():
        raise FileNotFoundError(feature_schema_uri)
    return ModelArtifact(
        model_id=str(payload["model_id"]),
        model_type=str(payload["model_type"]),
        feature_set_id=str(payload["feature_set_id"]),
        label_name=str(payload["label_name"]),
        label_base=str(payload["label_base"]),
        horizon_d=int(payload["horizon_d"]),
        feature_schema_uri=feature_schema_uri,
        artifact_uri=artifact_uri,
        artifact_dir=Path(str(payload.get("artifact_dir", artifact_uri.parent))),
        metrics={key: float(value) for key, value in dict(payload.get("metrics") or {}).items()},
    )


def predict_daily_alpha_risk(
    con,
    feature_mart: pd.DataFrame,
    alpha_artifact: ModelArtifact,
    risk_artifact: ModelArtifact,
    *,
    run_id: str,
    fold_id: str,
    score_version: str,
) -> pd.DataFrame:
    absolute_score = predict_with_model(feature_mart, alpha_artifact)
    risk_prob = predict_with_model(feature_mart, risk_artifact)
    now = datetime.now(timezone.utc).isoformat()
    raw = feature_mart[["trade_date", "code"]].copy()
    raw["run_id"] = run_id
    raw["fold_id"] = fold_id
    raw["score_version"] = score_version
    raw["feature_set_id"] = alpha_artifact.feature_set_id
    raw["horizon_d"] = alpha_artifact.horizon_d
    raw["absolute_model_id"] = alpha_artifact.model_id
    raw["active_model_id"] = None
    raw["risk_model_id"] = risk_artifact.model_id
    raw["absolute_score"] = list(absolute_score)
    raw["active_score"] = None
    raw["risk_prob"] = list(risk_prob)
    raw["generated_at"] = now
    upsert_dataframe(con, "ml_prediction_raw_daily", raw, ["trade_date", "code", "run_id", "fold_id", "horizon_d"])

    predictions = alpha_risk_predictions_from_raw(raw)
    upsert_dataframe(con, "ml_predictions_daily", predictions, ["trade_date", "code", "model_id", "horizon_d"])
    return predictions


def write_predictions_to_live_db(predictions: pd.DataFrame, live_db: str | Path | None) -> int:
    if live_db is None or predictions.empty:
        return 0
    live_db_path = Path(live_db)
    snapshot_dir = (
        live_db_path.parent
        / "artifacts"
        / PROFIT_PROTECT_PORTFOLIO_ID
        / PROFIT_PROTECT_SCORE_VERSION
        / PROFIT_PROTECT_SERVING_FOLD_ID
    )
    con = init_live_sim_db(live_db_path)
    try:
        bundle = activate_profit_protect_live_bundle(con, artifact_snapshot_dir=snapshot_dir)
        written = upsert_live_predictions(con, predictions, bundle_id=str(bundle["bundle_id"]))
        return int(len(written))
    finally:
        con.close()


def alpha_risk_predictions_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out["absolute_rank_pct"] = _percent_rank(out["absolute_score"])
    out["risk_rank_pct"] = _percent_rank(out["risk_prob"])
    out["model_id"] = "alpha_risk:" + out["absolute_model_id"].astype(str) + ":" + out["risk_model_id"].astype(str)
    out["alpha_score"] = out["absolute_score"]
    out["alpha_rank_pct"] = out["absolute_rank_pct"]
    out["reg_score"] = None
    out["active_score"] = out["absolute_score"]
    out["active_rank_pct"] = out["absolute_rank_pct"]
    out["risk_score"] = out["risk_prob"]
    out["context_score"] = None
    out["liquidity_score"] = None
    out["relative_strength_pct"] = 0.5
    out["resonance_pct"] = 0.5
    out["penalty_score"] = 0.0
    out["core_score"] = out["absolute_rank_pct"]
    out["trade_score"] = None
    out["trade_score_v2"] = out["absolute_rank_pct"]
    columns = [
        "trade_date",
        "code",
        "model_id",
        "horizon_d",
        "alpha_score",
        "alpha_rank_pct",
        "absolute_score",
        "absolute_rank_pct",
        "reg_score",
        "active_score",
        "active_rank_pct",
        "risk_score",
        "risk_prob",
        "risk_rank_pct",
        "context_score",
        "liquidity_score",
        "relative_strength_pct",
        "resonance_pct",
        "penalty_score",
        "core_score",
        "trade_score",
        "trade_score_v2",
        "score_version",
        "run_id",
        "fold_id",
        "absolute_model_id",
        "active_model_id",
        "risk_model_id",
        "feature_set_id",
        "generated_at",
    ]
    return out[columns]


def _percent_rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    ranks = pd.Series(0.0, index=values.index, dtype=float)
    count = int(valid.sum())
    if count <= 1:
        return ranks
    ranks.loc[valid] = (numeric.loc[valid].rank(method="min", ascending=True) - 1.0) / float(count - 1)
    return ranks


if __name__ == "__main__":
    main()
