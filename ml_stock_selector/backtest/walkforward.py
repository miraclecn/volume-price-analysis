from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml_stock_selector.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from ml_stock_selector.backtest.execution import ExecutionConfig
from ml_stock_selector.constants import FEATURE_SET_BASELINE_A
from ml_stock_selector.models.alpha_ranker import train_alpha_ranker
from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.constructor import construct_portfolio_targets
from ml_stock_selector.prediction import build_prediction_rows, predict_with_model
from ml_stock_selector.sample_builder import build_training_samples
from ml_stock_selector.scoring import add_context_score, add_liquidity_score, score_candidates


@dataclass(frozen=True)
class WalkForwardFoldResult:
    fold_id: str
    model_ids: list[str]
    predictions: pd.DataFrame
    targets: pd.DataFrame
    backtest_result: BacktestResult
    metrics: dict[str, float]


def run_walkforward_experiment(
    config,
    normalized_bars: pd.DataFrame,
    feature_mart: pd.DataFrame,
    labels: pd.DataFrame,
    tradeability: pd.DataFrame,
    artifact_dir: Path | str = "outputs/ml/artifacts",
) -> list[WalkForwardFoldResult]:
    feature_set_id = str(config.features.get("feature_set_id", FEATURE_SET_BASELINE_A))
    if feature_set_id not in set(feature_mart["feature_set_id"]):
        feature_set_id = str(feature_mart["feature_set_id"].iloc[0])
    horizon = int(config.labels.get("main_horizon", 1))
    if horizon not in set(labels["horizon_d"]):
        horizon = int(labels["horizon_d"].iloc[0])
    samples = build_training_samples(feature_mart, labels, feature_set_id, horizon, str(config.labels.get("label_base", "from_next_open")))
    if samples.empty:
        samples = build_training_samples(feature_mart, labels, feature_set_id, horizon, "from_close")
    artifact = train_alpha_ranker(samples, feature_set_id, "rank_label", str(samples["label_base"].iloc[0]), horizon, artifact_dir)
    scores = predict_with_model(feature_mart, artifact)
    predictions = build_prediction_rows(feature_mart, scores, artifact)
    predictions = predictions.merge(feature_mart[["trade_date", "code", "industry_code", "is_st", "is_paused", "adv20_amount", "can_buy_next_open"]], on=["trade_date", "code"], how="left")
    predictions = score_candidates(add_liquidity_score(add_context_score(predictions)))
    targets = construct_portfolio_targets(predictions, PortfolioConstraints(min_trade_score=-999.0), "walkforward")
    targets = allocate_weights(targets, 0.05, 0.10, True)
    backtest = run_backtest(targets, normalized_bars, BacktestConfig(1000000.0, "walkforward", ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0)))
    return [WalkForwardFoldResult("fold_1", [artifact.model_id], predictions, targets, backtest, {"rows": float(len(predictions))})]

