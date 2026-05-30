from __future__ import annotations

import pandas as pd

from ml_stock_selector.backtest.metrics import (
    annualized_return,
    max_drawdown,
    ndcg_at_k,
    rank_ic,
    unknown_industry_daily_exposure,
)


def test_backtest_metrics_match_small_fixture():
    nav = pd.DataFrame({"nav": [1.0, 1.1, 1.0]})
    preds = pd.DataFrame({"trade_date": ["d"] * 3, "alpha_score": [0.9, 0.5, 0.1], "future_score": [3.0, 2.0, 1.0]})

    assert max_drawdown(nav) < 0
    assert annualized_return(nav) != 0
    assert rank_ic(preds, "alpha_score", "future_score") > 0
    assert ndcg_at_k(preds, "alpha_score", "future_score", 2) > 0


def test_unknown_industry_daily_exposure_counts_positions_and_weight():
    positions = pd.DataFrame(
        {
            "sim_date": ["2024-01-03", "2024-01-03"],
            "code": ["u1", "i1"],
            "industry_code": ["UNKNOWN", "I1"],
            "weight": [0.2, 0.3],
        }
    )

    exposure = unknown_industry_daily_exposure(positions)

    assert exposure["unknown_industry_position_count"].tolist() == [1]
    assert exposure["unknown_industry_weight"].tolist() == [0.2]
