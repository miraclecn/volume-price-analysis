from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd

from ml_stock_selector.constants import MODEL_TYPE_ACTIVE_RANKER
from ml_stock_selector.models.alpha_ranker import LoadedAlphaRanker, train_alpha_ranker
from ml_stock_selector.models.artifacts import ModelArtifact
from ml_stock_selector.models.config import LightGBMRankerConfig


def train_active_ranker(
    samples: pd.DataFrame,
    feature_set_id: str,
    label_name: str,
    label_base: str,
    horizon_d: int,
    artifact_dir: Path | str,
    deny_industry: bool = False,
    train_config: LightGBMRankerConfig | None = None,
) -> ModelArtifact:
    artifact = train_alpha_ranker(
        samples,
        feature_set_id,
        label_name,
        label_base,
        horizon_d,
        artifact_dir,
        deny_industry=deny_industry,
        train_config=train_config,
    )
    model_id = artifact.model_id.replace("alpha_ranker", "active_ranker", 1)
    _copy_params_file(artifact, model_id)
    return ModelArtifact(
        model_id,
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


def _copy_params_file(artifact: ModelArtifact, model_id: str) -> None:
    source = artifact.artifact_uri.with_suffix(".params.json")
    if source.exists():
        shutil.copy2(source, artifact.artifact_dir / f"{model_id}.params.json")
