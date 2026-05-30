from __future__ import annotations

from pathlib import Path

from ml_stock_selector.models.artifacts import ModelArtifact
from ml_stock_selector.registry import get_active_model


def load_active_model(
    con,
    model_type: str,
    feature_set_id: str,
    label_name: str,
    label_base: str,
    horizon_d: int,
) -> ModelArtifact:
    row = get_active_model(con, model_type, feature_set_id, label_name, label_base, horizon_d)
    artifact_uri = Path(str(row["artifact_uri"]))
    return ModelArtifact(
        model_id=str(row["model_id"]),
        model_type=str(row["model_type"]),
        feature_set_id=str(row["feature_set_id"]),
        label_name=str(row["label_name"]),
        label_base=str(row["label_base"]),
        horizon_d=int(row["horizon_d"]),
        feature_schema_uri=Path(str(row["feature_schema_uri"])),
        artifact_uri=artifact_uri,
        artifact_dir=artifact_uri.parent,
        metrics={},
    )

