from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.constants import MODEL_TYPE_ACTIVE_RANKER, MODEL_TYPE_RANKER, MODEL_TYPE_RISK, SCORE_VERSION_THREE_MODEL
from ml_stock_selector.models.config import artifact_params_json
from ml_stock_selector.registry import register_model, register_model_bundle
from ml_stock_selector.runtime.run_context import create_run_context, register_run_context, update_run_status
from ml_stock_selector.storage import init_ml_db
from ml_stock_selector.universe import apply_universe_filter
from scripts.train_ml_models import train_model_artifacts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bundle-id")
    parser.add_argument("--bundle-role", default="production")
    parser.add_argument("--feature-set-id")
    parser.add_argument("--score-version", default=SCORE_VERSION_THREE_MODEL)
    parser.add_argument("--run-artifact-dir", default="outputs/ml/runs")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_ml_config(args.config)
    feature_set_id = args.feature_set_id or str(config.features["feature_set_id"])
    horizon = int(config.labels["main_horizon"])
    label_base = str(config.labels["label_base"])
    ml_db = str(config.data["ml_db"])
    bundle_id = args.bundle_id or args.run_id
    con = init_ml_db(ml_db)
    try:
        context = create_run_context(
            run_type="production_train",
            run_id=args.run_id,
            experiment_name=f"{args.bundle_role}_bundle",
            config_path=args.config,
            artifact_root=args.run_artifact_dir,
            alpha_data_db=str(config.data["alpha_data_db"]),
            ml_db=ml_db,
            feature_set_id=feature_set_id,
            label_version=f"{label_base}_h{horizon}",
            score_version=args.score_version,
        )
        register_run_context(con, context)
        feature_mart = con.execute("select * from ml_feature_mart_daily where feature_set_id = ?", [feature_set_id]).fetchdf()
        feature_mart = apply_universe_filter(feature_mart, exclude_bse=bool(config.universe.get("exclude_bse", False)))
        labels = con.execute("select * from ml_labels_daily").fetchdf()
        artifacts = train_model_artifacts(
            feature_mart,
            labels,
            feature_set_id,
            horizon,
            label_base,
            context.artifact_root / "models",
            config.ml_v2,
            config.model,
            bool(config.universe.get("exclude_bse", False)),
            config.portfolio.get("v2", {}).get("min_adv20_amount", config.portfolio.get("min_adv20_amount")),
        )
        by_type = {artifact.model_type: artifact for artifact in artifacts}
        required = {MODEL_TYPE_RANKER, MODEL_TYPE_ACTIVE_RANKER, MODEL_TYPE_RISK}
        missing = required - set(by_type)
        if missing:
            raise ValueError(f"production bundle training requires three model roles, missing: {', '.join(sorted(missing))}")
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
                params_json=artifact_params_json(artifact),
                metrics_json=json.dumps(artifact.metrics, sort_keys=True),
                run_id=args.run_id,
            )
        register_model_bundle(
            con,
            bundle_id=bundle_id,
            run_id=args.run_id,
            bundle_role=args.bundle_role,
            absolute_model_id=by_type[MODEL_TYPE_RANKER].model_id,
            active_model_id=by_type[MODEL_TYPE_ACTIVE_RANKER].model_id,
            risk_model_id=by_type[MODEL_TYPE_RISK].model_id,
            feature_set_id=feature_set_id,
            label_base=label_base,
            horizon_d=horizon,
            score_version=args.score_version,
            artifact_dir=str(context.artifact_root),
            status="candidate",
        )
        update_run_status(con, context, "success")
    finally:
        con.close()
    print(f"bundle_id={bundle_id}")


if __name__ == "__main__":
    main()
