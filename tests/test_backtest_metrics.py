from __future__ import annotations

import pandas as pd

from ml_stock_selector.backtest.metrics import annualized_return, max_drawdown, ndcg_at_k, rank_ic


def test_backtest_metrics_match_small_fixture():
    nav = pd.DataFrame({"nav": [1.0, 1.1, 1.0]})
    preds = pd.DataFrame({"trade_date": ["d"] * 3, "alpha_score": [0.9, 0.5, 0.1], "future_score": [3.0, 2.0, 1.0]})

    assert max_drawdown(nav) < 0
    assert annualized_return(nav) != 0
    assert rank_ic(preds, "alpha_score", "future_score") > 0
    assert ndcg_at_k(preds, "alpha_score", "future_score", 2) > 0

