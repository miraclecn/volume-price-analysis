from __future__ import annotations

import pandas as pd

from ml_stock_selector.prediction import rank_raw_predictions_sql
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


def test_sql_prediction_ranking_computes_percent_ranks_and_trade_score(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    raw = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "run_id": "run",
                "fold_id": "wf",
                "score_version": "v2_three_model",
                "feature_set_id": "vpa_d_sequence",
                "horizon_d": 5,
                "absolute_model_id": "abs",
                "active_model_id": "active",
                "risk_model_id": "risk",
                "absolute_score": 1.0,
                "active_score": 1.0,
                "risk_prob": 0.2,
                "generated_at": "t",
            },
            {
                "trade_date": "2024-01-02",
                "code": "000002.SZ",
                "run_id": "run",
                "fold_id": "wf",
                "score_version": "v2_three_model",
                "feature_set_id": "vpa_d_sequence",
                "horizon_d": 5,
                "absolute_model_id": "abs",
                "active_model_id": "active",
                "risk_model_id": "risk",
                "absolute_score": 2.0,
                "active_score": 2.0,
                "risk_prob": 0.8,
                "generated_at": "t",
            },
        ]
    )
    upsert_dataframe(con, "ml_prediction_raw_daily", raw, ["trade_date", "code", "run_id", "fold_id", "horizon_d"])

    rank_raw_predictions_sql(con, "run", "wf", "v2_three_model")

    rows = con.execute(
        """
        select code, absolute_rank_pct, active_rank_pct, risk_rank_pct, trade_score_v2
        from ml_predictions_daily
        order by code
        """
    ).fetchall()
    assert rows[0] == ("000001.SZ", 0.0, 0.0, 0.0, 0.0)
    assert rows[1] == ("000002.SZ", 1.0, 1.0, 1.0, 0.65)

