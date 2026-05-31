from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.models.active_ranker import train_active_ranker
from ml_stock_selector.models.alpha_ranker import train_alpha_ranker
from ml_stock_selector.models.risk_model import train_risk_model
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
        artifacts = train_model_artifacts(
            feature_mart,
            labels,
            feature_set_id,
            horizon,
            label_base,
            Path(str(config.data["artifact_dir"])),
            config.ml_v2,
        )
        for artifact in artifacts:
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
    print("model_ids=" + ",".join(artifact.model_id for artifact in artifacts))


def train_model_artifacts(
    feature_mart,
    labels,
    feature_set_id: str,
    horizon: int,
    label_base: str,
    artifact_dir: Path,
    ml_v2: dict[str, object],
):
    artifacts = []
    deny_industry = bool(ml_v2.get("feature_matrix_v2_deny_industry"))
    absolute_label = "absolute_label" if bool(ml_v2.get("labels_v2_enabled")) and "absolute_label" in labels else "rank_label"
    absolute_samples = build_training_samples(
        feature_mart,
        labels,
        feature_set_id,
        horizon,
        label_base,
        label_name=absolute_label,
    )
    artifacts.append(train_alpha_ranker(absolute_samples, feature_set_id, absolute_label, label_base, horizon, artifact_dir, deny_industry=deny_industry))
    if bool(ml_v2.get("active_ranker_enabled")):
        active_samples = build_training_samples(
            feature_mart,
            labels,
            feature_set_id,
            horizon,
            label_base,
            label_name="active_label",
        )
        artifacts.append(train_active_ranker(active_samples, feature_set_id, "active_label", label_base, horizon, artifact_dir, deny_industry=deny_industry))
    if bool(ml_v2.get("risk_model_v2_enabled")):
        risk_samples = build_training_samples(
            feature_mart,
            labels,
            feature_set_id,
            horizon,
            label_base,
            label_name="risk_label",
        )
        artifacts.append(train_risk_model(risk_samples, feature_set_id, "risk_label", label_base, horizon, artifact_dir, deny_industry=deny_industry))
    return artifacts


if __name__ == "__main__":
    main()
