from __future__ import annotations

import json

import duckdb
import pandas as pd

from ml_stock_selector.backtest.metrics import (
    summarize_unknown_industry_exposure,
    unknown_industry_daily_exposure,
)
from ml_stock_selector.backtest.reports import unknown_industry_report_metrics
from ml_stock_selector.constants import FEATURE_SET_BASELINE_A
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.feature_matrix import build_feature_matrix
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.models.alpha_ranker import train_alpha_ranker
from ml_stock_selector.portfolio.constraints import PortfolioConstraints, apply_hard_filters
from ml_stock_selector.portfolio.constructor import construct_portfolio_targets
from ml_stock_selector.registry import activate_model, register_model
from ml_stock_selector.sample_builder import build_training_samples
from ml_stock_selector.serving.daily_signal import generate_daily_signal
from ml_stock_selector.storage import init_ml_db, upsert_dataframe
from ml_stock_selector.tradeability import build_tradeability_mart
from tests.ml_fixtures import create_vpa_db, normalized_bars


def unknown_bars() -> pd.DataFrame:
    bars = normalized_bars()
    mask = bars["code"] == "000004.SZ"
    bars.loc[mask, "industry_code"] = "UNKNOWN"
    bars.loc[mask, "industry_name"] = "UNKNOWN"
    bars.loc[mask, "data_quality_usability_flags"] = "MISSING_INDUSTRY_CODE"
    return bars


def test_feature_mart_and_samples_keep_unknown_industry(tmp_path):
    bars = unknown_bars()
    feature_mart = build_feature_mart(
        str(create_vpa_db(tmp_path / "vpa.duckdb")),
        bars,
        "2024-01-02",
        "2024-01-08",
        FEATURE_SET_BASELINE_A,
        [5],
        build_tradeability_mart(bars),
    )
    labels = build_labels(bars, [1])
    samples = build_training_samples(
        feature_mart,
        labels,
        FEATURE_SET_BASELINE_A,
        1,
        "from_next_open",
    )

    unknown_rows = feature_mart[feature_mart["industry_code"] == "UNKNOWN"]
    unknown_features = json.loads(unknown_rows.iloc[0]["features_json"])

    assert not unknown_rows.empty
    assert unknown_features["industry_code"] == "UNKNOWN"
    assert unknown_features["industry_name"] == "UNKNOWN"
    assert unknown_features["industry_unknown"] is True
    assert "000004.SZ" in set(samples["code"])


def test_feature_matrix_encodes_unknown_industry_as_stable_category():
    train = pd.DataFrame(
        {
            "features_json": [
                json.dumps({"industry_code": "I1", "x": 1.0}),
                json.dumps({"industry_code": "UNKNOWN", "x": 2.0}),
            ]
        }
    )
    matrix, schema = build_feature_matrix(train, FEATURE_SET_BASELINE_A, fit=True)
    inference = pd.DataFrame({"features_json": [json.dumps({"industry_code": "UNKNOWN", "x": 3.0})]})
    inferred, _ = build_feature_matrix(inference, FEATURE_SET_BASELINE_A, schema=schema)

    assert "UNKNOWN" in schema.category_levels["industry_code"]
    assert matrix["industry_code=UNKNOWN"].sum() == 1.0
    assert list(inferred.columns) == schema.output_columns
    assert inferred.iloc[0]["industry_code=UNKNOWN"] == 1.0


def test_feature_matrix_maps_unseen_unknown_to_unknown_bucket():
    train = pd.DataFrame({"features_json": [json.dumps({"industry_code": "I1", "x": 1.0})]})
    _, schema = build_feature_matrix(train, FEATURE_SET_BASELINE_A, fit=True)
    inference = pd.DataFrame({"features_json": [json.dumps({"industry_code": "UNKNOWN", "x": 3.0})]})

    inferred, _ = build_feature_matrix(inference, FEATURE_SET_BASELINE_A, schema=schema)

    assert inferred.iloc[0]["industry_code=__UNKNOWN__"] == 1.0


def _portfolio_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-02"] * 5,
            "code": ["u1", "u2", "i1a", "i1b", "i2a"],
            "industry_code": ["UNKNOWN", "UNKNOWN", "I1", "I1", "I2"],
            "industry_name": ["UNKNOWN", "UNKNOWN", "Industry 1", "Industry 1", "Industry 2"],
            "trade_score": [0.99, 0.98, 0.97, 0.96, 0.95],
            "is_st": [False] * 5,
            "is_paused": [False] * 5,
            "can_buy_next_open": [True] * 5,
            "adv20_amount": [100.0] * 5,
        }
    )


def test_portfolio_limits_unknown_industry_independently():
    constraints = PortfolioConstraints(
        target_positions=4,
        hard_max_positions=4,
        max_industry_names=1,
        max_unknown_industry_names=1,
        max_new_entries_per_day=5,
        min_trade_score=0.0,
    )

    targets = construct_portfolio_targets(_portfolio_candidates(), constraints, "p1")

    assert targets[targets["industry_code"] == "UNKNOWN"]["code"].tolist() == ["u1"]
    assert set(targets["industry_code"]) == {"UNKNOWN", "I1", "I2"}
    assert "u2" not in set(targets["code"])


def test_portfolio_can_disable_unknown_industry():
    constraints = PortfolioConstraints(
        target_positions=4,
        hard_max_positions=4,
        max_unknown_industry_names=0,
        max_new_entries_per_day=5,
        min_trade_score=0.0,
    )

    targets = construct_portfolio_targets(_portfolio_candidates(), constraints, "p1")

    assert "UNKNOWN" not in set(targets["industry_code"])


def test_unknown_industry_still_respects_hard_filters():
    candidates = _portfolio_candidates()
    candidates.loc[candidates["code"] == "u1", "is_st"] = True

    filtered = apply_hard_filters(candidates, PortfolioConstraints(min_trade_score=0.0))
    targets = construct_portfolio_targets(
        filtered,
        PortfolioConstraints(max_unknown_industry_names=1, min_trade_score=0.0),
        "p1",
    )

    assert "u1" not in set(filtered["code"])
    assert targets[targets["industry_code"] == "UNKNOWN"]["code"].tolist() == ["u2"]


def test_backtest_reports_unknown_industry_exposure():
    positions = pd.DataFrame(
        {
            "sim_date": ["2024-01-03", "2024-01-03", "2024-01-04"],
            "code": ["u1", "i1", "u1"],
            "industry_code": ["UNKNOWN", "I1", "UNKNOWN"],
            "weight": [0.10, 0.20, 0.15],
        }
    )
    orders = pd.DataFrame(
        {
            "sim_date": ["2024-01-03", "2024-01-04"],
            "code": ["u1", "i1"],
            "industry_code": ["UNKNOWN", "I1"],
            "status": ["filled", "filled"],
        }
    )

    daily = unknown_industry_daily_exposure(positions, orders)
    summary = summarize_unknown_industry_exposure(daily)
    report_metrics = unknown_industry_report_metrics(daily)

    assert daily.loc[daily["sim_date"] == "2024-01-03", "unknown_industry_position_count"].iloc[0] == 1
    assert daily.loc[daily["sim_date"] == "2024-01-03", "unknown_industry_weight"].iloc[0] == 0.10
    assert daily["unknown_industry_trade_count"].sum() == 1
    assert summary["unknown_industry_max_weight"] == 0.15
    assert report_metrics["unknown_industry_days"] == 2.0


def test_daily_signal_marks_unknown_selection_and_limit(tmp_path):
    bars = unknown_bars()
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

    con = init_ml_db(tmp_path / "ml.duckdb")
    upsert_dataframe(con, "ml_feature_mart_daily", feature_mart, ["trade_date", "code", "feature_set_id"])
    register_model(
        con,
        model_id=artifact.model_id,
        model_type=artifact.model_type,
        feature_set_id=artifact.feature_set_id,
        label_name=artifact.label_name,
        label_base=artifact.label_base,
        horizon_d=artifact.horizon_d,
        artifact_uri=str(artifact.artifact_uri),
        feature_schema_uri=str(artifact.feature_schema_uri),
    )
    activate_model(con, artifact.model_id)

    _, selected_targets = generate_daily_signal(
        con,
        "2024-01-02",
        FEATURE_SET_BASELINE_A,
        1,
        "p_selected",
        PortfolioConstraints(target_positions=4, hard_max_positions=4, max_unknown_industry_names=1, min_trade_score=-999.0),
    )
    predictions, targets = generate_daily_signal(
        con,
        "2024-01-02",
        FEATURE_SET_BASELINE_A,
        1,
        "p1",
        PortfolioConstraints(target_positions=4, hard_max_positions=4, max_unknown_industry_names=0, min_trade_score=-999.0),
    )
    con.close()

    selected_unknown = selected_targets[selected_targets["industry_code"] == "UNKNOWN"]
    unknown_predictions = predictions[predictions["industry_code"] == "UNKNOWN"]
    assert "industry_unknown" in selected_unknown.iloc[0]["entry_reason"]
    assert not unknown_predictions.empty
    assert "industry_name" in predictions.columns
    assert "unknown_industry_limit" in set(unknown_predictions["exclusion_reason"])
    assert "UNKNOWN" not in set(targets.get("industry_code", pd.Series(dtype=object)))
