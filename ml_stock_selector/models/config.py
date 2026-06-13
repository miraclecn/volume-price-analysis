from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any


@dataclass(frozen=True)
class LightGBMRankerConfig:
    objective: str = "lambdarank"
    metric: str = "ndcg"
    n_estimators: int = 25
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_data_in_leaf: int = 1
    feature_fraction: float | None = None
    bagging_fraction: float | None = None
    bagging_freq: int | None = None
    lambda_l2: float | None = None
    random_state: int | None = None
    eval_at: tuple[int, ...] = (10, 15)
    lambdarank_truncation_level: int | None = None
    early_stopping_rounds: int = 0
    num_threads: int | None = None
    force_col_wise: bool | None = None
    max_bin: int | None = None
    histogram_pool_size: int | None = None

    def to_params(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["eval_at"] = list(self.eval_at)
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True)
class LightGBMRiskConfig:
    objective: str = "binary"
    n_estimators: int = 25
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_data_in_leaf: int = 1
    feature_fraction: float | None = None
    bagging_fraction: float | None = None
    bagging_freq: int | None = None
    lambda_l2: float | None = None
    class_weight: str | None = None
    random_state: int | None = None
    early_stopping_rounds: int = 0
    num_threads: int | None = None
    force_col_wise: bool | None = None
    max_bin: int | None = None
    histogram_pool_size: int | None = None

    def to_params(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def ranker_config_from_model_section(model: dict[str, object] | None) -> LightGBMRankerConfig:
    raw = _merged_lightgbm_config(model, "alpha_ranker")
    fields = LightGBMRankerConfig.__dataclass_fields__
    return LightGBMRankerConfig(**{key: _coerce_config_value(key, raw[key]) for key in fields if key in raw})


def risk_config_from_model_section(model: dict[str, object] | None) -> LightGBMRiskConfig:
    raw = _merged_lightgbm_config(model, "risk_model")
    fields = LightGBMRiskConfig.__dataclass_fields__
    return LightGBMRiskConfig(**{key: _coerce_config_value(key, raw[key]) for key in fields if key in raw})


def _merged_lightgbm_config(model: dict[str, object] | None, section: str) -> dict[str, object]:
    model = model or {}
    raw: dict[str, object] = {}
    if isinstance(model.get(section), dict):
        raw.update(model[section])
    if isinstance(model.get("lightgbm_runtime"), dict):
        raw.update(model["lightgbm_runtime"])
    return raw


def _coerce_config_value(key: str, value: object) -> object:
    if key == "eval_at" and not isinstance(value, tuple):
        return tuple(int(item) for item in value)  # type: ignore[arg-type]
    return value


def artifact_params_json(artifact) -> str:
    path = artifact.artifact_uri.with_suffix(".params.json")
    if not path.exists():
        path = artifact.artifact_dir / f"{artifact.model_id}.params.json"
    if not path.exists():
        return "{}"
    return json.dumps(json.loads(path.read_text(encoding="utf-8")), sort_keys=True)
