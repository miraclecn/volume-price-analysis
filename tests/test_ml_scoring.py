from __future__ import annotations

import pandas as pd

from ml_stock_selector.models.calibrator import cross_sectional_percentile
from ml_stock_selector.scoring import add_context_score, add_liquidity_score, score_candidates


def test_scoring_defaults_and_risk_penalty():
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-02"],
            "code": ["a", "b"],
            "alpha_score": [0.1, 0.9],
            "risk_score": [0.1, 0.9],
            "amount": [10.0, 20.0],
        }
    )

    scored = score_candidates(add_liquidity_score(add_context_score(frame)))

    assert scored["context_score"].tolist() == [0.5, 0.5]
    assert scored.loc[scored["code"] == "b", "trade_score"].iloc[0] > scored.loc[scored["code"] == "a", "trade_score"].iloc[0]
    assert cross_sectional_percentile(frame, "alpha_score").between(0, 1).all()

