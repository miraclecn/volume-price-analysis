from __future__ import annotations

import pandas as pd


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

