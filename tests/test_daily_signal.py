from __future__ import annotations

import duckdb
import pandas as pd
from types import SimpleNamespace

from ml_stock_selector.constants import FEATURE_SET_BASELINE_A, SCORE_VERSION_THREE_MODEL
from ml_stock_selector.feature_mart import build_feature_mart
from ml_stock_selector.label_builder import build_labels
from ml_stock_selector.models.alpha_ranker import train_alpha_ranker
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.holding_policy import HoldingPolicy
from ml_stock_selector.registry import activate_model, register_model
from ml_stock_selector.sample_builder import build_training_samples
from ml_stock_selector.serving.artifact_loader import load_active_model
from ml_stock_selector.serving.daily_signal import generate_daily_signal
from ml_stock_selector.storage import init_ml_db, upsert_dataframe
from ml_stock_selector.tradeability import build_tradeability_mart
from scripts.train_ml_models import train_model_artifacts
from tests.ml_fixtures import create_vpa_db, normalized_bars


def test_daily_signal_loads_active_model_and_writes_predictions(tmp_path):
    bars = normalized_bars()
    tradeability = build_tradeability_mart(bars)
    feature_mart = build_feature_mart(str(create_vpa_db(tmp_path / "vpa.duckdb")), bars, "2024-01-02", "2024-01-08", FEATURE_SET_BASELINE_A, [5], tradeability)
    labels = build_labels(bars, [1])
    samples = build_training_samples(feature_mart, labels, FEATURE_SET_BASELINE_A, 1, "from_next_open")
    artifact = train_alpha_ranker(samples, FEATURE_SET_BASELINE_A, "rank_label", "from_next_open", 1, tmp_path)

    con = init_ml_db(tmp_path / "ml.duckdb")
    upsert_dataframe(con, "ml_feature_mart_daily", feature_mart, ["trade_date", "code", "feature_set_id"])
    register_model(con, model_id=artifact.model_id, model_type=artifact.model_type, feature_set_id=artifact.feature_set_id, label_name=artifact.label_name, label_base=artifact.label_base, horizon_d=artifact.horizon_d, artifact_uri=str(artifact.artifact_uri), feature_schema_uri=str(artifact.feature_schema_uri))
    activate_model(con, artifact.model_id)

    loaded = load_active_model(con, artifact.model_type, FEATURE_SET_BASELINE_A, "rank_label", "from_next_open", 1)
    predictions, targets = generate_daily_signal(con, "2024-01-03", FEATURE_SET_BASELINE_A, 1, "p1")

    assert loaded.model_id == artifact.model_id
    assert not predictions.empty
    assert "industry_name" in predictions.columns
    assert "entry_reason" in targets.columns
    assert len(targets) <= 15
    con.close()


def test_daily_signal_v2_loads_three_models_and_writes_v2_scores(tmp_path):
    bars = normalized_bars()
    tradeability = build_tradeability_mart(bars)
    feature_mart = build_feature_mart(str(create_vpa_db(tmp_path / "vpa.duckdb")), bars, "2024-01-02", "2024-01-08", FEATURE_SET_BASELINE_A, [5], tradeability)
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

    con = init_ml_db(tmp_path / "ml.duckdb")
    upsert_dataframe(con, "ml_feature_mart_daily", feature_mart, ["trade_date", "code", "feature_set_id"])
    for artifact in artifacts:
        register_model(con, model_id=artifact.model_id, model_type=artifact.model_type, feature_set_id=artifact.feature_set_id, label_name=artifact.label_name, label_base=artifact.label_base, horizon_d=artifact.horizon_d, artifact_uri=str(artifact.artifact_uri), feature_schema_uri=str(artifact.feature_schema_uri))
        activate_model(con, artifact.model_id)

    predictions, targets = generate_daily_signal(
        con,
        "2024-01-03",
        FEATURE_SET_BASELINE_A,
        1,
        "p_v2",
        PortfolioConstraints(
            target_positions=3,
            hard_max_positions=3,
            max_new_entries_per_day=3,
            min_trade_score=0.0,
            candidate_min_count=1,
            candidate_absolute_min_rank_pct=0.0,
            candidate_active_min_rank_pct=0.0,
            candidate_risk_max_rank_pct=1.0,
            core_absolute_min_rank_pct=0.0,
            core_active_min_rank_pct=0.0,
            core_risk_max_rank_pct=1.0,
            core_min_trade_score=-999.0,
            holding_policy=HoldingPolicy(sell_score_threshold=-1.0),
        ),
        use_v2=True,
    )
    con.close()

    assert not predictions.empty
    assert predictions["score_version"].eq(SCORE_VERSION_THREE_MODEL).all()
    assert {"absolute_score", "active_score", "risk_prob", "core_score", "trade_score_v2"}.issubset(predictions.columns)
    assert {"entry_reason", "hold_reason", "exit_reason", "sell_blocked_reason"}.issubset(targets.columns)
    assert {"run_id", "fold_id", "score_version"}.issubset(targets.columns)
    assert targets["score_version"].eq(SCORE_VERSION_THREE_MODEL).all()


def test_daily_signal_v2_passes_current_holdings_to_portfolio_constructor(monkeypatch, tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    current_holdings = pd.DataFrame({"code": ["held"]})
    feature_mart = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "held",
                "feature_set_id": FEATURE_SET_BASELINE_A,
                "industry_code": "I1",
                "industry_name": "Industry 1",
                "is_st": False,
                "is_paused": False,
                "adv20_amount": 100000000.0,
                "can_buy_next_open": True,
                "can_sell_next_open": True,
                "is_bse": False,
            },
            {
                "trade_date": "2024-01-02",
                "code": "new",
                "feature_set_id": FEATURE_SET_BASELINE_A,
                "industry_code": "I2",
                "industry_name": "Industry 2",
                "is_st": False,
                "is_paused": False,
                "adv20_amount": 100000000.0,
                "can_buy_next_open": True,
                "can_sell_next_open": True,
                "is_bse": False,
            },
        ]
    )
    artifacts = {
        "absolute_label": SimpleNamespace(model_id="abs", model_type="alpha_ranker", feature_set_id=FEATURE_SET_BASELINE_A, horizon_d=1),
        "active_label": SimpleNamespace(model_id="act", model_type="active_ranker", feature_set_id=FEATURE_SET_BASELINE_A, horizon_d=1),
        "risk_label": SimpleNamespace(model_id="risk", model_type="risk_model", feature_set_id=FEATURE_SET_BASELINE_A, horizon_d=1),
    }
    captured = {}

    monkeypatch.setattr(
        "ml_stock_selector.serving.daily_signal.load_active_model",
        lambda _con, _model_type, _feature_set_id, label_name, _label_base, _horizon_d: artifacts[label_name],
    )
    monkeypatch.setattr(
        "ml_stock_selector.serving.daily_signal._load_daily_features",
        lambda _con, _as_of_date, _feature_set_id, _feature_store_spec: feature_mart,
    )
    monkeypatch.setattr(
        "ml_stock_selector.serving.daily_signal.predict_with_model",
        lambda frame, artifact: pd.Series([0.9, 0.8] if artifact.model_id != "risk" else [0.1, 0.2], index=frame.index),
    )

    def fake_construct(predictions, constraints, portfolio_id, *, current_holdings, **_metadata):
        captured["current_holdings"] = current_holdings.copy()
        return pd.DataFrame(
            [
                {
                    "trade_date": "2024-01-02",
                    "portfolio_id": portfolio_id,
                    "code": "held",
                    "target_weight": 0.0,
                    "rank_n": 1,
                    "trade_score": 0.9,
                    "entry_reason": "core_pool",
                    "generated_at": "t",
                }
            ]
        )

    monkeypatch.setattr(
        "ml_stock_selector.serving.daily_signal.construct_portfolio_targets_v2",
        fake_construct,
    )

    generate_daily_signal(
        con,
        "2024-01-02",
        FEATURE_SET_BASELINE_A,
        1,
        "p_v2",
        PortfolioConstraints(
            target_positions=1,
            hard_max_positions=1,
            max_initial_entries=1,
            max_new_entries_per_day=1,
            min_adv20_amount=0,
            min_candidate_pool_size=1,
            candidate_min_trade_score=0.0,
            core_min_trade_score=0.0,
            holding_policy=HoldingPolicy(sell_score_threshold=-1.0),
        ),
        use_v2=True,
        current_holdings=current_holdings,
    )
    con.close()

    assert captured["current_holdings"]["code"].tolist() == ["held"]
