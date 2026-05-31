from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.constants import MODEL_TYPE_ACTIVE_RANKER, MODEL_TYPE_RANKER, MODEL_TYPE_RISK
from ml_stock_selector.prediction import build_prediction_rows, build_three_model_prediction_rows, predict_with_model, upsert_predictions
from ml_stock_selector.registry import get_active_model
from ml_stock_selector.serving.artifact_loader import load_active_model
from ml_stock_selector.storage import init_ml_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--model-id")
    args = parser.parse_args()
    config = load_ml_config(args.config)
    con = init_ml_db(str(config.data["ml_db"]))
    try:
        feature_set_id = str(config.features["feature_set_id"])
        horizon = int(config.labels["main_horizon"])
        if args.model_id:
            row = con.execute("select * from ml_model_registry where model_id = ?", [args.model_id]).fetchdf().iloc[0].to_dict()
            from ml_stock_selector.models.artifacts import ModelArtifact
            artifact = ModelArtifact(row["model_id"], row["model_type"], row["feature_set_id"], row["label_name"], row["label_base"], int(row["horizon_d"]), Path(row["feature_schema_uri"]), Path(row["artifact_uri"]), Path(row["artifact_uri"]).parent, {})
        else:
            label_name = "absolute_label" if bool(config.ml_v2["labels_v2_enabled"]) else "rank_label"
            artifact = load_active_model(con, MODEL_TYPE_RANKER, feature_set_id, label_name, str(config.labels["label_base"]), horizon)
        feature_mart = con.execute("select * from ml_feature_mart_daily where feature_set_id = ?", [artifact.feature_set_id]).fetchdf()
        if bool(config.ml_v2["active_ranker_enabled"]) and bool(config.ml_v2["risk_model_v2_enabled"]) and not args.model_id:
            active = load_active_model(con, MODEL_TYPE_ACTIVE_RANKER, feature_set_id, "active_label", str(config.labels["label_base"]), horizon)
            risk = load_active_model(con, MODEL_TYPE_RISK, feature_set_id, "risk_label", str(config.labels["label_base"]), horizon)
            rows = build_three_model_prediction_rows(
                feature_mart,
                predict_with_model(feature_mart, artifact),
                predict_with_model(feature_mart, active),
                predict_with_model(feature_mart, risk),
                artifact,
                active,
                risk,
            )
        else:
            rows = build_prediction_rows(feature_mart, predict_with_model(feature_mart, artifact), artifact)
        upsert_predictions(con, rows)
    finally:
        con.close()
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
