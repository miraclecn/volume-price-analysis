from __future__ import annotations

import pickle

import pandas as pd

from ml_stock_selector.constants import MODEL_TYPE_ACTIVE_RANKER, MODEL_TYPE_RANKER, MODEL_TYPE_RISK
from ml_stock_selector.feature_matrix import FeatureSchema, save_feature_schema
from ml_stock_selector.feature_store import write_feature_frame_to_feature_store
from ml_stock_selector.feature_store_reader import FeatureStoreSpec
from ml_stock_selector.models.alpha_ranker import LinearFallbackModel
from ml_stock_selector.models.risk_model import LogisticFallbackModel
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.registry import activate_model_bundle, register_model, register_model_bundle
from ml_stock_selector.serving.daily_signal import generate_daily_signal
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


def test_daily_signal_can_read_as_of_date_from_feature_store_without_feature_json_table(tmp_path):
    feature_mart = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "feature_set_id": "vpa_d_sequence",
                "features_json": '{"ret_1d": 0.4}',
            },
            {
                "trade_date": "2024-01-02",
                "code": "920001.BJ",
                "feature_set_id": "vpa_d_sequence",
                "features_json": '{"ret_1d": 0.9}',
            },
        ]
    )
    write_feature_frame_to_feature_store(
        feature_mart,
        output_dir=tmp_path / "feature_store",
        dataset_version="v2",
        feature_set_id="vpa_d_sequence",
        source_db="test",
    )
    con = init_ml_db(tmp_path / "ml.duckdb")
    tradeability = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "industry_code": "I1",
                "industry_name": "Industry",
                "is_st": False,
                "is_paused": False,
                "adv20_amount": 100000000.0,
                "can_buy_next_open": True,
                "can_sell_next_open": True,
                "is_bse": False,
                "generated_at": "t",
            },
            {
                "trade_date": "2024-01-02",
                "code": "920001.BJ",
                "industry_code": "I2",
                "industry_name": "BSE",
                "is_st": False,
                "is_paused": False,
                "adv20_amount": 100000000.0,
                "can_buy_next_open": True,
                "can_sell_next_open": True,
                "is_bse": True,
                "generated_at": "t",
            },
        ]
    )
    upsert_dataframe(con, "ml_tradeability_daily", tradeability, ["trade_date", "code"])
    schema_path = tmp_path / "schema.json"
    save_feature_schema(
        FeatureSchema(
            feature_set_id="vpa_d_sequence",
            numeric_columns=["ret_1d"],
            categorical_columns=[],
            output_columns=["ret_1d"],
            category_levels={},
            fill_values={"ret_1d": 0.0},
            schema_version="v2",
        ),
        schema_path,
    )
    artifacts = {
        ("abs", MODEL_TYPE_RANKER, "absolute_label"): LinearFallbackModel(["ret_1d"], {"ret_1d": 1.0}),
        ("act", MODEL_TYPE_ACTIVE_RANKER, "active_label"): LinearFallbackModel(["ret_1d"], {"ret_1d": 1.0}),
        ("risk", MODEL_TYPE_RISK, "risk_label"): LogisticFallbackModel(["ret_1d"], {"ret_1d": -1.0}),
    }
    for (model_id, model_type, label_name), model in artifacts.items():
        artifact_path = tmp_path / f"{model_id}.pkl"
        with artifact_path.open("wb") as handle:
            pickle.dump(model, handle)
        register_model(
            con,
            model_id=model_id,
            model_type=model_type,
            feature_set_id="vpa_d_sequence",
            label_name=label_name,
            label_base="from_next_open",
            horizon_d=5,
            artifact_uri=str(artifact_path),
            feature_schema_uri=str(schema_path),
        )
    register_model_bundle(
        con,
        bundle_id="bundle_feature_store",
        run_id="run_feature_store",
        bundle_role="production",
        absolute_model_id="abs",
        active_model_id="act",
        risk_model_id="risk",
        feature_set_id="vpa_d_sequence",
        label_base="from_next_open",
        horizon_d=5,
        score_version="v2_three_model",
        artifact_dir=str(tmp_path / "bundle_feature_store"),
    )
    activate_model_bundle(con, "bundle_feature_store")

    predictions, targets = generate_daily_signal(
        con,
        "2024-01-02",
        "vpa_d_sequence",
        5,
        "p_v2",
        PortfolioConstraints(
            target_positions=1,
            hard_max_positions=1,
            max_new_entries_per_day=1,
            min_adv20_amount=0,
            min_trade_score=-999.0,
            candidate_min_count=1,
            candidate_absolute_min_rank_pct=0.0,
            candidate_active_min_rank_pct=0.0,
            candidate_risk_max_rank_pct=1.0,
            core_absolute_min_rank_pct=0.0,
            core_active_min_rank_pct=0.0,
            core_risk_max_rank_pct=1.0,
            core_min_trade_score=-999.0,
        ),
        use_v2=True,
        exclude_bse=True,
        feature_store_spec=FeatureStoreSpec(str(tmp_path / "feature_store"), "v2", "vpa_d_sequence"),
    )
    con.close()

    assert predictions["code"].tolist() == ["000001.SZ"]
    assert "features_json" not in predictions.columns
    assert "can_buy_next_open" in predictions.columns
    assert len(targets) == 1
