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

