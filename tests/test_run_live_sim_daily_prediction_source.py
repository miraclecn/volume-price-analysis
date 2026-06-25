from __future__ import annotations

import duckdb
import pandas as pd

from ml_stock_selector.serving.live_sim import activate_profit_protect_live_bundle, init_live_sim_db, upsert_live_predictions
from scripts.run_live_sim_daily import _load_live_predictions_with_tradeability


def test_load_live_predictions_with_tradeability_uses_live_prediction_table_and_shared_tradeability(tmp_path):
    live_con = init_live_sim_db(tmp_path / "live.duckdb")
    bundle = activate_profit_protect_live_bundle(live_con)
    upsert_live_predictions(
        live_con,
        pd.DataFrame(
            [
                {
                    "trade_date": "2026-06-16",
                    "code": "000001.SZ",
                    "model_id": "alpha_risk:a:r",
                    "horizon_d": 5,
                    "absolute_score": 0.1,
                    "absolute_rank_pct": 0.9,
                    "risk_prob": 0.2,
                    "risk_rank_pct": 0.1,
                    "trade_score_v2": 0.8,
                    "adv20_amount": 20_000_000.0,
                    "generated_at": "now",
                }
            ]
        ),
        bundle_id=str(bundle["bundle_id"]),
    )
    ml_con = duckdb.connect(":memory:")
    ml_con.execute(
        """
        create table ml_tradeability_daily (
            trade_date varchar,
            code varchar,
            industry_code varchar,
            industry_name varchar,
            is_st boolean,
            is_paused boolean,
            is_bse boolean,
            adv20_amount double,
            can_buy_next_open boolean,
            can_sell_next_open boolean,
            next_open double,
            next_limit_up double,
            next_limit_down double,
            next_is_paused boolean,
            limit_up_pct double,
            limit_down_pct double,
            limit_band varchar
        )
        """
    )
    ml_con.execute(
        """
        insert into ml_tradeability_daily values
        ('2026-06-16', '000001.SZ', 'I1', 'Industry', false, false, false, 30000000,
         true, true, 10, 11, 9, false, 0.1, -0.1, '10')
        """
    )

    out = _load_live_predictions_with_tradeability(
        live_con,
        ml_con,
        "2026-06-16",
        str(bundle["bundle_id"]),
    )

    assert out["code"].tolist() == ["000001.SZ"]
    assert out.iloc[0]["industry_code"] == "I1"
    assert out.iloc[0]["adv20_amount"] == 30_000_000.0
    assert out.iloc[0]["absolute_rank_pct"] == 0.9
    live_con.close()
    ml_con.close()
