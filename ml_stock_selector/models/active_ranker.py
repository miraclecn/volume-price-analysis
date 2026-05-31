from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_stock_selector.constants import MODEL_TYPE_ACTIVE_RANKER
from ml_stock_selector.models.alpha_ranker import LoadedAlphaRanker, train_alpha_ranker
from ml_stock_selector.models.artifacts import ModelArtifact


def train_active_ranker(
    samples: pd.DataFrame,
    feature_set_id: str,
    label_name: str,
    label_base: str,
    horizon_d: int,
    artifact_dir: Path | str,
    deny_industry: bool = False,
) -> ModelArtifact:
    artifact = train_alpha_ranker(samples, feature_set_id, label_name, label_base, horizon_d, artifact_dir, deny_industry=deny_industry)
    return ModelArtifact(
        artifact.model_id.replace("alpha_ranker", "active_ranker", 1),
        MODEL_TYPE_ACTIVE_RANKER,
        artifact.feature_set_id,
        artifact.label_name,
        artifact.label_base,
        artifact.horizon_d,
        artifact.feature_schema_uri,
        artifact.artifact_uri,
        artifact.artifact_dir,
        artifact.metrics,
    )


def load_active_ranker(artifact: ModelArtifact) -> LoadedAlphaRanker:
    return LoadedAlphaRanker(artifact)
