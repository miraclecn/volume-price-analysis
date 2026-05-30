from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.models.alpha_ranker import train_alpha_ranker
from ml_stock_selector.registry import activate_model, register_model
from ml_stock_selector.sample_builder import build_training_samples
from ml_stock_selector.storage import init_ml_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--feature-set-id")
    args = parser.parse_args()
    config = load_ml_config(args.config)
    feature_set_id = args.feature_set_id or str(config.features["feature_set_id"])
    horizon = int(config.labels["main_horizon"])
    label_base = str(config.labels["label_base"])
    con = init_ml_db(str(config.data["ml_db"]))
    try:
        feature_mart = con.execute("select * from ml_feature_mart_daily where feature_set_id = ?", [feature_set_id]).fetchdf()
        labels = con.execute("select * from ml_labels_daily").fetchdf()
        samples = build_training_samples(feature_mart, labels, feature_set_id, horizon, label_base)
        artifact = train_alpha_ranker(samples, feature_set_id, "rank_label", label_base, horizon, Path(str(config.data["artifact_dir"])))
        register_model(
            con,
            model_id=artifact.model_id,
            model_type=artifact.model_type,
            feature_set_id=artifact.feature_set_id,
            label_name=artifact.label_name,
            label_base=artifact.label_base,
            horizon_d=artifact.horizon_d,
            artifact_uri=str(artifact.artifact_uri),
            feature_schema_uri=str(artifact.feature_schema_uri),
        )
        activate_model(con, artifact.model_id)
    finally:
        con.close()
    print(f"model_id={artifact.model_id}")


if __name__ == "__main__":
    main()
