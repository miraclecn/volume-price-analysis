from __future__ import annotations

from ml_stock_selector.constants import MODEL_TYPE_RANKER
from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.constructor import construct_portfolio_targets
from ml_stock_selector.prediction import build_prediction_rows, predict_with_model
from ml_stock_selector.scoring import add_context_score, add_liquidity_score, score_candidates
from ml_stock_selector.serving.artifact_loader import load_active_model
from ml_stock_selector.storage import upsert_dataframe


def generate_daily_signal(
    con,
    as_of_date: str,
    feature_set_id: str,
    horizon_d: int,
    portfolio_id: str,
):
    artifact = load_active_model(con, MODEL_TYPE_RANKER, feature_set_id, "rank_label", "from_next_open", horizon_d)
    feature_mart = con.execute(
        """
        select *
        from ml_feature_mart_daily
        where trade_date = ? and feature_set_id = ?
        order by code
        """,
        [as_of_date, feature_set_id],
    ).fetchdf()
    scores = predict_with_model(feature_mart, artifact)
    predictions = build_prediction_rows(feature_mart, scores, artifact)
    enrich_cols = ["trade_date", "code", "industry_code", "is_st", "is_paused", "adv20_amount", "can_buy_next_open"]
    predictions = predictions.merge(feature_mart[enrich_cols], on=["trade_date", "code"], how="left")
    predictions = score_candidates(add_liquidity_score(add_context_score(predictions)))
    targets = construct_portfolio_targets(predictions, PortfolioConstraints(min_trade_score=-999.0), portfolio_id)
    targets = allocate_weights(targets, 0.05, 0.10, allow_cash=True)
    upsert_dataframe(con, "ml_predictions_daily", predictions, ["trade_date", "code", "model_id", "horizon_d"])
    upsert_dataframe(con, "ml_portfolio_targets_daily", targets, ["trade_date", "portfolio_id", "code"])
    return predictions, targets

