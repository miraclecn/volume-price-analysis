from __future__ import annotations

from pathlib import Path

from ml_stock_selector.models.artifacts import ModelArtifact
from ml_stock_selector.registry import get_active_model, get_active_model_bundle


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


def load_model_by_id(con, model_id: str) -> ModelArtifact:
    row = con.execute("select * from ml_model_registry where model_id = ?", [model_id]).fetchdf()
    if row.empty:
        raise ValueError(f"Unknown model_id: {model_id}")
    return _artifact_from_row(row.iloc[0].to_dict())


def load_active_model_bundle(
    con,
    *,
    bundle_role: str,
    feature_set_id: str,
    label_base: str,
    horizon_d: int,
) -> dict[str, ModelArtifact]:
    bundle = get_active_model_bundle(con, bundle_role, feature_set_id, label_base, horizon_d)
    return {
        "absolute": load_model_by_id(con, str(bundle["absolute_model_id"])),
        "active": load_model_by_id(con, str(bundle["active_model_id"])),
        "risk": load_model_by_id(con, str(bundle["risk_model_id"])),
    }


def _artifact_from_row(row: dict[str, object]) -> ModelArtifact:
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
