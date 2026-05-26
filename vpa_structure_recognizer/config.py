from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class VPAConfig:
    windows: list[int]
    parent_windows: dict[int, list[int]]
    volume_levels: dict[str, float]
    price_position: dict[str, float]
    label_thresholds: dict[str, float]
    scoring_weights: dict[str, float]
    rating_thresholds: dict[str, int]
    sources: dict[str, str]
    outputs: dict[str, str]


def load_config(path: Path | str) -> VPAConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    scoring = raw["scoring"]
    return VPAConfig(
        windows=[int(window) for window in raw["windows"]["base"]],
        parent_windows={
            int(window): [int(parent) for parent in parents]
            for window, parents in raw["windows"]["parent_map"].items()
        },
        volume_levels={key: float(value) for key, value in raw["volume_levels"].items()},
        price_position={key: float(value) for key, value in raw["price_position"].items()},
        label_thresholds={
            key: float(value) for key, value in raw["label_thresholds"].items()
        },
        scoring_weights={
            "market": float(scoring["market_weight"]),
            "sector": float(scoring["sector_weight"]),
            "stock": float(scoring["stock_weight"]),
            "resonance": float(scoring["resonance_weight"]),
        },
        rating_thresholds={key: int(value) for key, value in raw["rating"].items()},
        sources={key: str(value) for key, value in raw["sources"].items()},
        outputs={key: str(value) for key, value in raw["outputs"].items()},
    )
