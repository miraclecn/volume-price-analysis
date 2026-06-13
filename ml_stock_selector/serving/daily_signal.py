from __future__ import annotations

import pickle

import pandas as pd

from ml_stock_selector.constants import (
    MODEL_TYPE_ACTIVE_RANKER,
    MODEL_TYPE_RANKER,
    MODEL_TYPE_RISK,
    SCORE_VERSION_LEGACY,
    SCORE_VERSION_THREE_MODEL,
)
from ml_stock_selector.feature_matrix import load_feature_schema
from ml_stock_selector.feature_store_reader import FeatureStoreSpec, iter_feature_store_batches
from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constraints import (
    PortfolioConstraints,
    apply_hard_filters,
    is_unknown_industry,
)
from ml_stock_selector.portfolio.constructor import construct_portfolio_targets, construct_portfolio_targets_v2
from ml_stock_selector.portfolio.constructor import get_portfolio_diagnostics
from ml_stock_selector.prediction import build_prediction_rows, build_three_model_prediction_rows, predict_with_model
from ml_stock_selector.scoring import add_context_score, add_liquidity_score, score_candidates, score_candidates_v2
from ml_stock_selector.serving.artifact_loader import load_active_model, load_active_model_bundle
from ml_stock_selector.storage import upsert_dataframe
from ml_stock_selector.universe import apply_universe_filter


def generate_daily_signal(
    con,
    as_of_date: str,
    feature_set_id: str,
    horizon_d: int,
    portfolio_id: str,
    constraints: PortfolioConstraints | None = None,
    use_v2: bool = False,
    exclude_bse: bool = False,
    feature_store_spec: FeatureStoreSpec | None = None,
    current_holdings: pd.DataFrame | None = None,
    run_id: str = "daily_signal",
    fold_id: str = "daily",
    score_version: str | None = None,
):
    constraints = constraints or (PortfolioConstraints() if use_v2 else PortfolioConstraints(min_trade_score=-999.0))
    score_version = score_version or (SCORE_VERSION_THREE_MODEL if use_v2 else SCORE_VERSION_LEGACY)
    rank_label = "absolute_label" if use_v2 else "rank_label"
    if use_v2:
        bundle = load_active_model_bundle(
            con,
            bundle_role="production",
            feature_set_id=feature_set_id,
            label_base="from_next_open",
            horizon_d=horizon_d,
        )
        artifact = bundle["absolute"]
    else:
        artifact = load_active_model(con, MODEL_TYPE_RANKER, feature_set_id, rank_label, "from_next_open", horizon_d)
    feature_mart = _load_daily_features(con, as_of_date, feature_set_id, feature_store_spec)
    feature_mart = apply_universe_filter(feature_mart, exclude_bse=exclude_bse)
    if use_v2:
        active = bundle["active"]
        risk = bundle["risk"]
        if feature_store_spec is not None:
            absolute_scores = _predict_matrix_with_artifact(feature_mart, artifact, proba=False)
            active_scores = _predict_matrix_with_artifact(feature_mart, active, proba=False)
            risk_scores = _predict_matrix_with_artifact(feature_mart, risk, proba=True)
        else:
            absolute_scores = predict_with_model(feature_mart, artifact)
            active_scores = predict_with_model(feature_mart, active)
            risk_scores = predict_with_model(feature_mart, risk)
        predictions = build_three_model_prediction_rows(
            feature_mart,
            absolute_scores,
            active_scores,
            risk_scores,
            artifact,
            active,
            risk,
        )
        predictions["absolute_model_id"] = artifact.model_id
        predictions["active_model_id"] = active.model_id
        predictions["risk_model_id"] = risk.model_id
    else:
        scores = predict_with_model(feature_mart, artifact)
        predictions = build_prediction_rows(feature_mart, scores, artifact)
        predictions["absolute_model_id"] = artifact.model_id
        predictions["active_model_id"] = None
        predictions["risk_model_id"] = None
    enrich_cols = [
        "trade_date",
        "code",
        "industry_code",
        "industry_name",
        "is_st",
        "is_paused",
        "adv20_amount",
        "can_buy_next_open",
        "can_sell_next_open",
        "is_bse",
    ]
    predictions = predictions.merge(feature_mart[enrich_cols], on=["trade_date", "code"], how="left")
    if use_v2:
        predictions = score_candidates_v2(add_liquidity_score(add_context_score(predictions)))
        hard_filtered = apply_hard_filters(predictions, constraints, score_column="trade_score_v2")
        targets = construct_portfolio_targets_v2(
            predictions,
            constraints,
            portfolio_id,
            current_holdings=current_holdings,
            run_id=run_id,
            fold_id=fold_id,
            score_version=score_version,
        )
    else:
        predictions = score_candidates(add_liquidity_score(add_context_score(predictions)))
        hard_filtered = apply_hard_filters(predictions, constraints)
        targets = construct_portfolio_targets(predictions, constraints, portfolio_id)
    diagnostics = get_portfolio_diagnostics(targets)
    targets = allocate_weights(targets, 0.05, 0.10, allow_cash=True)
    predictions["run_id"] = run_id
    predictions["fold_id"] = fold_id
    predictions["score_version"] = score_version
    targets = _annotate_targets(targets, run_id, fold_id, portfolio_id, score_version)
    diagnostics = _annotate_diagnostics(diagnostics, run_id, fold_id, portfolio_id, score_version)
    predictions = _annotate_selection_reasons(predictions, hard_filtered, targets, constraints)
    upsert_dataframe(con, "ml_predictions_daily", predictions, ["trade_date", "code", "model_id", "horizon_d"])
    upsert_dataframe(con, "ml_portfolio_targets_daily", targets, ["trade_date", "run_id", "fold_id", "portfolio_id", "score_version", "code"])
    upsert_dataframe(con, "ml_portfolio_construction_diagnostics", diagnostics, ["trade_date", "run_id", "fold_id", "portfolio_id", "score_version"])
    return predictions, targets


def _annotate_targets(
    targets: pd.DataFrame,
    run_id: str,
    fold_id: str,
    portfolio_id: str,
    score_version: str,
) -> pd.DataFrame:
    if targets.empty:
        return targets
    out = targets.copy()
    out["run_id"] = run_id
    out["fold_id"] = fold_id
    out["portfolio_id"] = portfolio_id
    out["score_version"] = score_version
    return out


def _annotate_diagnostics(
    diagnostics: pd.DataFrame,
    run_id: str,
    fold_id: str,
    portfolio_id: str,
    score_version: str,
) -> pd.DataFrame:
    if diagnostics.empty:
        return diagnostics
    out = diagnostics.copy()
    out["run_id"] = run_id
    out["fold_id"] = fold_id
    out["portfolio_id"] = portfolio_id
    out["score_version"] = score_version
    return out


def _load_daily_features(
    con,
    as_of_date: str,
    feature_set_id: str,
    feature_store_spec: FeatureStoreSpec | None,
):
    if feature_store_spec is None:
        return con.execute(
            """
            select *
            from ml_feature_mart_daily
            where trade_date = ? and feature_set_id = ?
            order by code
            """,
            [as_of_date, feature_set_id],
        ).fetchdf()
    batches = list(
        iter_feature_store_batches(
            feature_store_spec,
            as_of_date,
            as_of_date,
            columns=None,
            batch_size=50000,
        )
    )
    features = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(columns=["trade_date", "code"])
    if features.empty:
        return features
    features["feature_set_id"] = feature_set_id
    tradeability = con.execute(
        """
        select trade_date, code, industry_code, industry_name, is_st, is_paused,
               adv20_amount, can_buy_next_open, can_sell_next_open, is_bse
        from ml_tradeability_daily
        where trade_date = ?
        """,
        [as_of_date],
    ).fetchdf()
    return features.merge(tradeability, on=["trade_date", "code"], how="left").sort_values("code").reset_index(drop=True)


def _predict_matrix_with_artifact(features, artifact, *, proba: bool):
    schema = load_feature_schema(artifact.feature_schema_uri)
    matrix = pd.DataFrame(index=features.index)
    for column in schema.output_columns:
        values = features[column] if column in features else pd.Series(0.0, index=features.index)
        matrix[column] = pd.to_numeric(values, errors="coerce").fillna(0.0)
    with artifact.artifact_uri.open("rb") as handle:
        model = pickle.load(handle)
    if proba:
        return model.predict_proba_matrix(matrix[schema.output_columns])
    return model.predict_matrix(matrix[schema.output_columns])


def _annotate_selection_reasons(
    predictions,
    hard_filtered,
    targets,
    constraints: PortfolioConstraints,
):
    output = predictions.copy()
    selected = set(targets["code"]) if not targets.empty and "code" in targets else set()
    hard_filter_passed = set(hard_filtered["code"]) if not hard_filtered.empty else set()
    output["exclusion_reason"] = ""
    unknown_mask = output["industry_code"].map(is_unknown_industry)
    limited_unknown = (
        unknown_mask
        & ~output["code"].isin(selected)
        & output["code"].isin(hard_filter_passed)
        & (constraints.max_unknown_industry_names >= 0)
    )
    output.loc[limited_unknown, "exclusion_reason"] = "unknown_industry_limit"
    return output
