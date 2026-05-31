from __future__ import annotations

import json

from ml_stock_selector.backtest.engine import BacktestConfig, run_backtest
from ml_stock_selector.backtest.execution import ExecutionConfig
from ml_stock_selector.constants import FEATURE_SET_BASELINE_A, MODEL_TYPE_ACTIVE_RANKER, MODEL_TYPE_RANKER, MODEL_TYPE_RISK, SCORE_VERSION_THREE_MODEL
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.models.alpha_ranker import train_alpha_ranker
from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.constructor import construct_portfolio_targets, construct_portfolio_targets_v2
from ml_stock_selector.prediction import build_prediction_rows, build_three_model_prediction_rows, predict_with_model
from ml_stock_selector.sample_builder import build_training_samples
from ml_stock_selector.scoring import add_context_score, add_liquidity_score, score_candidates, score_candidates_v2
from ml_stock_selector.tradeability import build_tradeability_mart
from scripts.train_ml_models import train_model_artifacts
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


def test_ml_pipeline_v2_smoke_runs_without_industry_features(tmp_path):
    bars = normalized_bars()
    bars.loc[bars["code"] == "000004.SZ", "industry_code"] = "UNKNOWN"
    bars.loc[bars["code"] == "000004.SZ", "industry_name"] = "UNKNOWN"
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
    artifacts = train_model_artifacts(
        feature_mart,
        labels,
        FEATURE_SET_BASELINE_A,
        1,
        "from_next_open",
        tmp_path,
        {"labels_v2_enabled": True, "active_ranker_enabled": True, "risk_model_v2_enabled": True},
    )
    by_type = {artifact.model_type: artifact for artifact in artifacts}
    predictions = build_three_model_prediction_rows(
        feature_mart,
        predict_with_model(feature_mart, by_type[MODEL_TYPE_RANKER]),
        predict_with_model(feature_mart, by_type[MODEL_TYPE_ACTIVE_RANKER]),
        predict_with_model(feature_mart, by_type[MODEL_TYPE_RISK]),
        by_type[MODEL_TYPE_RANKER],
        by_type[MODEL_TYPE_ACTIVE_RANKER],
        by_type[MODEL_TYPE_RISK],
    )
    predictions = predictions.merge(
        feature_mart[["trade_date", "code", "industry_code", "industry_name", "is_st", "is_paused", "adv20_amount", "can_buy_next_open"]],
        on=["trade_date", "code"],
        how="left",
    )
    scored = score_candidates_v2(add_liquidity_score(add_context_score(predictions)))
    targets = allocate_weights(
        construct_portfolio_targets_v2(
            scored,
            PortfolioConstraints(
                min_trade_score=0.0,
                candidate_min_count=1,
                candidate_absolute_min_rank_pct=0.0,
                candidate_active_min_rank_pct=0.0,
                candidate_risk_max_rank_pct=1.0,
                core_absolute_min_rank_pct=0.0,
                core_active_min_rank_pct=0.0,
                core_risk_max_rank_pct=1.0,
                core_min_trade_score=-999.0,
            ),
            "smoke_v2",
        ),
        0.05,
        0.10,
        True,
    )

    first_features = json.loads(feature_mart.iloc[0]["features_json"])
    assert "industry_code" not in first_features
    assert labels["active_label"].notna().all()
    assert predictions["score_version"].eq(SCORE_VERSION_THREE_MODEL).all()
    assert {"absolute_score", "active_score", "risk_prob", "trade_score_v2"}.issubset(scored.columns)
    assert targets.empty or targets["target_weight"].sum() <= 1.0
