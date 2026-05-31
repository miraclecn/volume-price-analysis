from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class MLConfig:
    data: dict[str, object]
    features: dict[str, object]
    labels: dict[str, object]
    split: dict[str, object]
    model: dict[str, object]
    portfolio: dict[str, object]
    backtest: dict[str, object]
    ml_v2: dict[str, object]


DEFAULT_ML_V2_CONFIG: dict[str, object] = {
    "exclude_industry_metadata_from_features_json": False,
    "feature_matrix_v2_deny_industry": False,
    "labels_v2_enabled": False,
    "active_ranker_enabled": False,
    "risk_model_v2_enabled": False,
    "trade_score_v2_enabled": False,
    "daily_signal_v2_enabled": False,
    "candidate_absolute_min_rank_pct": 0.70,
    "candidate_active_min_rank_pct": 0.70,
    "candidate_risk_max_rank_pct": 0.60,
    "core_absolute_min_rank_pct": 0.80,
    "core_active_min_rank_pct": 0.75,
    "core_risk_max_rank_pct": 0.35,
    "core_min_trade_score": 0.80,
}


def load_ml_config(path: Path | str) -> MLConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    required = ["data", "features", "labels", "split", "model", "portfolio", "backtest"]
    missing = [section for section in required if section not in raw]
    if missing:
        raise ValueError(f"Missing ML config sections: {', '.join(missing)}")
    config = MLConfig(
        data=raw["data"],
        features=raw["features"],
        labels=raw["labels"],
        split=raw["split"],
        model=raw["model"],
        portfolio=raw["portfolio"],
        backtest=raw["backtest"],
        ml_v2={**DEFAULT_ML_V2_CONFIG, **raw.get("ml_v2", {})},
    )
    _validate_config(config)
    return config


def _validate_config(config: MLConfig) -> None:
    portfolio = config.portfolio
    if portfolio["target_positions"] > portfolio["hard_max_positions"]:
        raise ValueError("target_positions cannot exceed hard_max_positions")
    if portfolio["single_name_min_weight"] > portfolio["single_name_max_weight"]:
        raise ValueError("single_name_min_weight cannot exceed single_name_max_weight")
    if config.backtest["a_share_lot_size"] <= 0:
        raise ValueError("a_share_lot_size must be positive")
