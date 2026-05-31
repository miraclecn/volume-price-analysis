from __future__ import annotations

import pandas as pd

from ml_stock_selector.constants import SCORE_VERSION_THREE_MODEL
from ml_stock_selector.models.calibrator import cross_sectional_percentile
from ml_stock_selector.scoring import add_context_score, add_liquidity_score, score_candidates, score_candidates_v2


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


def test_v2_scoring_combines_absolute_active_and_risk():
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"] * 3,
            "code": ["abs_only", "active_plus", "risky"],
            "absolute_rank_pct": [0.80, 0.80, 0.90],
            "active_rank_pct": [0.50, 0.90, 0.90],
            "risk_rank_pct": [0.20, 0.20, 0.90],
            "liquidity_score_pct": [0.50, 0.50, 0.50],
            "penalty_score": [0.0, 0.0, 0.0],
        }
    )

    scored = score_candidates_v2(frame)

    active_plus = scored.loc[scored["code"] == "active_plus", "trade_score_v2"].iloc[0]
    abs_only = scored.loc[scored["code"] == "abs_only", "trade_score_v2"].iloc[0]
    risky = scored.loc[scored["code"] == "risky", "trade_score_v2"].iloc[0]
    assert active_plus > abs_only
    assert risky < active_plus
    assert scored["score_version"].eq(SCORE_VERSION_THREE_MODEL).all()
    assert "core_score" in scored.columns
