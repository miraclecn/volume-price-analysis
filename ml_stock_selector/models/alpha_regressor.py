from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_stock_selector.constants import MODEL_TYPE_REGRESSOR
from ml_stock_selector.models.alpha_ranker import train_alpha_ranker
from ml_stock_selector.models.artifacts import ModelArtifact


def train_alpha_regressor(
    samples: pd.DataFrame,
    feature_set_id: str,
    label_name: str,
    label_base: str,
    horizon_d: int,
    artifact_dir: Path | str,
) -> ModelArtifact:
    artifact = train_alpha_ranker(samples, feature_set_id, label_name, label_base, horizon_d, artifact_dir)
    return ModelArtifact(
        artifact.model_id.replace("alpha_ranker", "alpha_regressor", 1),
        MODEL_TYPE_REGRESSOR,
        artifact.feature_set_id,
        artifact.label_name,
        artifact.label_base,
        artifact.horizon_d,
        artifact.feature_schema_uri,
        artifact.artifact_uri,
        artifact.artifact_dir,
        artifact.metrics,
    )

