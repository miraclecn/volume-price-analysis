from __future__ import annotations

import pandas as pd

from ml_stock_selector.backtest.data_access import load_backtest_candidates
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


def test_backtest_candidates_have_tradeability_columns():
    predictions = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"],
            "code": ["000001.SZ"],
            "trade_score_v2": [0.9],
            "adv20_amount": [1e8],
            "is_st": [False],
            "is_paused": [False],
            "can_buy_next_open": [True],
        }
    )
    required = {"adv20_amount", "is_st", "is_paused", "can_buy_next_open"}
    assert required.issubset(predictions.columns)


def test_backtest_candidates_join_tradeability_and_exclude_bse(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    predictions = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "model_id": "m",
                "horizon_d": 5,
                "run_id": "run",
                "fold_id": "wf_2020",
                "score_version": "v2_three_model",
                "feature_set_id": "vpa_d_sequence",
                "absolute_rank_pct": 0.9,
                "active_rank_pct": 0.8,
                "risk_rank_pct": 0.2,
                "trade_score_v2": 0.95,
                "generated_at": "t",
            },
            {
                "trade_date": "2024-01-02",
                "code": "920001.BJ",
                "model_id": "m",
                "horizon_d": 5,
                "run_id": "run",
                "fold_id": "wf_2020",
                "score_version": "v2_three_model",
                "feature_set_id": "vpa_d_sequence",
                "absolute_rank_pct": 0.9,
                "active_rank_pct": 0.8,
                "risk_rank_pct": 0.2,
                "trade_score_v2": 0.95,
                "generated_at": "t",
            },
        ]
    )
    tradeability = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "industry_code": "I1",
                "industry_name": "Industry",
                "is_st": False,
                "is_paused": False,
                "is_bse": False,
                "adv20_amount": 100000000.0,
                "next_open": 10.0,
                "next_limit_up": 11.0,
                "next_limit_down": 9.0,
                "next_is_paused": False,
                "can_buy_next_open": True,
                "can_sell_next_open": True,
                "generated_at": "t",
            },
            {
                "trade_date": "2024-01-02",
                "code": "920001.BJ",
                "industry_code": "I2",
                "industry_name": "BSE",
                "is_st": False,
                "is_paused": False,
                "is_bse": True,
                "adv20_amount": 100000000.0,
                "next_open": 10.0,
                "next_limit_up": 11.0,
                "next_limit_down": 9.0,
                "next_is_paused": False,
                "can_buy_next_open": True,
                "can_sell_next_open": True,
                "generated_at": "t",
            },
        ]
    )
    upsert_dataframe(con, "ml_predictions_daily", predictions, ["trade_date", "code", "model_id", "horizon_d"])
    upsert_dataframe(con, "ml_tradeability_daily", tradeability, ["trade_date", "code"])

    candidates = load_backtest_candidates(con, run_id="run", fold_id="wf_2020", score_version="v2_three_model", exclude_bse=True)
    con.close()

    assert candidates["code"].tolist() == ["000001.SZ"]
    assert {
        "industry_code",
        "industry_name",
        "is_st",
        "is_paused",
        "is_bse",
        "adv20_amount",
        "can_buy_next_open",
        "can_sell_next_open",
        "next_open",
        "next_limit_up",
        "next_limit_down",
        "next_is_paused",
    }.issubset(candidates.columns)
