from __future__ import annotations

from ml_stock_selector.backtest.engine import BacktestConfig, run_backtest
from ml_stock_selector.backtest.execution import ExecutionConfig
from ml_stock_selector.constants import FEATURE_SET_BASELINE_A
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.models.alpha_ranker import train_alpha_ranker
from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.constructor import construct_portfolio_targets
from ml_stock_selector.prediction import build_prediction_rows, predict_with_model
from ml_stock_selector.sample_builder import build_training_samples
from ml_stock_selector.scoring import add_context_score, add_liquidity_score, score_candidates
from ml_stock_selector.tradeability import build_tradeability_mart
from tests.ml_fixtures import create_vpa_db, normalized_bars


def test_ml_pipeline_smoke_runs_data_to_backtest(tmp_path):
    bars = normalized_bars()
    tradeability = build_tradeability_mart(bars)
    feature_mart = build_feature_mart(
        str(create_vpa_db(tmp_path / "vpa.duckdb")),
        bars,
        "2024-01-02",
        "2024-01-08",
        FEATURE_SET_BASELINE_A,
        [5],
        tradeability,
    )
    labels = build_labels(bars, [1])
    samples = build_training_samples(feature_mart, labels, FEATURE_SET_BASELINE_A, 1, "from_next_open")
    artifact = train_alpha_ranker(samples, FEATURE_SET_BASELINE_A, "rank_label", "from_next_open", 1, tmp_path)
    predictions = build_prediction_rows(feature_mart, predict_with_model(feature_mart, artifact), artifact)
    predictions = predictions.merge(
        feature_mart[["trade_date", "code", "industry_code", "is_st", "is_paused", "adv20_amount", "can_buy_next_open"]],
        on=["trade_date", "code"],
        how="left",
    )
    scored = score_candidates(add_liquidity_score(add_context_score(predictions)))
    targets = allocate_weights(
        construct_portfolio_targets(scored, PortfolioConstraints(min_trade_score=-999.0), "smoke"),
        0.05,
        0.10,
        True,
    )
    result = run_backtest(
        targets,
        bars,
        BacktestConfig(1000000.0, "smoke", ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0)),
    )

    assert not predictions.empty
    assert not targets.empty
    assert not result.nav.empty
    assert ((result.orders["status"] != "filled") | (result.orders["sim_date"] > result.orders["decision_date"])).all()
