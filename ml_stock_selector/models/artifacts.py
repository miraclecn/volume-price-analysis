from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    model_type: str
    feature_set_id: str
    label_name: str
    label_base: str
    horizon_d: int
    feature_schema_uri: Path
    artifact_uri: Path
    artifact_dir: Path
    metrics: dict[str, float]

