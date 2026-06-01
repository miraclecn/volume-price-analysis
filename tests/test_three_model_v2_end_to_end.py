from __future__ import annotations

from ml_stock_selector.backtest.engine import BacktestConfig, run_backtest
from ml_stock_selector.backtest.execution import ExecutionConfig
from ml_stock_selector.benchmarks import build_benchmark_tables
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.constants import FEATURE_SET_BASELINE_A
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.constructor import construct_portfolio_targets_v2
from ml_stock_selector.prediction import build_three_model_prediction_rows, predict_with_model
from ml_stock_selector.scoring import add_context_score, add_liquidity_score, score_candidates_v2
from ml_stock_selector.tradeability import build_tradeability_mart
from ml_stock_selector.universe import apply_universe_filter
from scripts.train_ml_models import train_model_artifacts
from tests.ml_fixtures import create_vpa_db, normalized_bars


def test_three_model_v2_end_to_end_smoke(tmp_path):
    bars = normalized_bars()
    bars.loc[len(bars)] = {
        **bars.iloc[0].to_dict(),
        "code": "430001.BJ",
        "trade_date": bars.iloc[0]["trade_date"],
    }
    tradeability = build_tradeability_mart(bars)
    feature_mart = build_feature_mart(
        str(create_vpa_db(tmp_path / "vpa.duckdb")),
        bars,
        "2024-01-02",
        "2024-01-08",
        FEATURE_SET_BASELINE_A,
        [5],
        tradeability,
        exclude_industry_metadata_from_features_json=True,
    )
    labels = build_labels(bars, [1], include_v2=True)
    market_bm, industry_bm = build_benchmark_tables(labels)
    assert not market_bm.empty
    assert not industry_bm.empty

    config = load_ml_config("config/ml_default.toml")
    feature_mart = apply_universe_filter(feature_mart, exclude_bse=bool(config.universe.get("exclude_bse", False)))
    artifacts = train_model_artifacts(
        feature_mart,
        labels,
        FEATURE_SET_BASELINE_A,
        1,
        "from_next_open",
        tmp_path,
        config.ml_v2,
        exclude_bse=True,
    )
    by_type = {artifact.model_type: artifact for artifact in artifacts}
    predictions = build_three_model_prediction_rows(
        feature_mart,
        predict_with_model(feature_mart, by_type["alpha_ranker"]),
        predict_with_model(feature_mart, by_type["active_ranker"]),
        predict_with_model(feature_mart, by_type["risk_model"]),
        by_type["alpha_ranker"],
        by_type["active_ranker"],
        by_type["risk_model"],
    )
    predictions = predictions.merge(
        feature_mart[["trade_date", "code", "industry_code", "industry_name", "is_st", "is_paused", "adv20_amount", "can_buy_next_open"]],
        on=["trade_date", "code"],
        how="left",
    )
    scored = score_candidates_v2(add_liquidity_score(add_context_score(predictions)))
    constraints = PortfolioConstraints(
        min_trade_score=0.0,
        min_adv20_amount=0.0,
        candidate_min_count=1,
        candidate_absolute_min_rank_pct=0.0,
        candidate_active_min_rank_pct=0.0,
        candidate_risk_max_rank_pct=1.0,
        core_absolute_min_rank_pct=0.0,
        core_active_min_rank_pct=0.0,
        core_risk_max_rank_pct=1.0,
        core_min_trade_score=-999.0,
    )
    targets = allocate_weights(construct_portfolio_targets_v2(scored, constraints, "v2_smoke"), 0.05, 0.10, True)
    result = run_backtest(
        targets,
        bars,
        BacktestConfig(1000000.0, "v2_smoke", ExecutionConfig(slippage_bps=0, commission_bps=0, stamp_duty_bps=0)),
    )

    assert scored["score_version"].eq("v2_three_model").all()
    assert {"absolute_score", "active_score", "risk_prob", "trade_score_v2"}.issubset(scored.columns)
    assert not scored["code"].str.endswith(".BJ").any()
    assert result.nav is not None
