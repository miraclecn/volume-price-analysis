from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from ml_stock_selector.label_builder import DEFAULT_FUTURE_SCORE_WEIGHTS, DEFAULT_RANK_LABEL_THRESHOLDS


@dataclass(frozen=True)
class MLConfig:
    data: dict[str, object]
    features: dict[str, object]
    labels: dict[str, object]
    split: dict[str, object]
    universe: dict[str, object]
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
    "candidate_risk_max_rank_pct": 0.65,
    "candidate_min_trade_score": 0.65,
    "min_candidate_pool_size": 5,
    "core_absolute_min_rank_pct": 0.75,
    "core_active_min_rank_pct": 0.65,
    "core_risk_max_rank_pct": 0.55,
    "core_min_trade_score": 0.75,
}

DEFAULT_LABEL_CONFIG: dict[str, object] = {
    "future_score_weights": DEFAULT_FUTURE_SCORE_WEIGHTS,
    "rank_label_thresholds": DEFAULT_RANK_LABEL_THRESHOLDS,
    "rank_group_by_limit_band": False,
}


def load_ml_config(path: Path | str) -> MLConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    required = ["data", "features", "labels", "split", "model", "portfolio", "backtest"]
    missing = [section for section in required if section not in raw]
    if missing:
        raise ValueError(f"Missing ML config sections: {', '.join(missing)}")
    labels = _merge_label_config(raw["labels"])
    config = MLConfig(
        data=raw["data"],
        features=raw["features"],
        labels=labels,
        split=raw["split"],
        universe=raw.get("universe", {"exclude_bse": False}),
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
    v2 = portfolio.get("v2", {})
    if isinstance(v2, dict):
        exit_config = v2.get("exit", {})
        if isinstance(exit_config, dict) and "sell_score_threshold" in exit_config:
            buy_threshold = float(v2.get("candidate_min_trade_score", config.ml_v2["candidate_min_trade_score"]))
            if float(exit_config["sell_score_threshold"]) >= buy_threshold:
                raise ValueError("sell_score_threshold must be below candidate_min_trade_score")
    if config.backtest["a_share_lot_size"] <= 0:
        raise ValueError("a_share_lot_size must be positive")


def _merge_label_config(raw_labels: dict[str, object]) -> dict[str, object]:
    labels = {**DEFAULT_LABEL_CONFIG, **raw_labels}
    raw_weights = raw_labels.get("future_score_weights", {})
    labels["future_score_weights"] = {
        **DEFAULT_FUTURE_SCORE_WEIGHTS,
        **(raw_weights if isinstance(raw_weights, dict) else {}),
    }
    labels["rank_label_thresholds"] = raw_labels.get("rank_label_thresholds", DEFAULT_RANK_LABEL_THRESHOLDS)
    labels["rank_group_by_limit_band"] = bool(raw_labels.get("rank_group_by_limit_band", DEFAULT_LABEL_CONFIG["rank_group_by_limit_band"]))
    return labels
