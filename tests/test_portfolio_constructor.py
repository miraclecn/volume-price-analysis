from __future__ import annotations

import pandas as pd

from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constraints import PortfolioConstraints, apply_hard_filters
from ml_stock_selector.portfolio.constructor import construct_portfolio_targets


def candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-02"] * 5,
            "code": ["a", "b", "c", "d", "e"],
            "industry_code": ["I1", "I1", "I1", "I2", "I2"],
            "trade_score": [0.99, 0.98, 0.97, 0.96, 0.20],
            "is_st": [False, False, True, False, False],
            "is_paused": [False, False, False, False, False],
            "can_buy_next_open": [True, True, True, True, True],
            "adv20_amount": [100.0] * 5,
        }
    )


def test_portfolio_filters_selects_and_allocates():
    constraints = PortfolioConstraints(target_positions=3, hard_max_positions=4, max_industry_names=2, max_new_entries_per_day=3, min_trade_score=0.8)
    filtered = apply_hard_filters(candidates(), constraints)
    targets = construct_portfolio_targets(filtered, constraints, "p1")
    weighted = allocate_weights(targets, 0.05, 0.5, allow_cash=True)

    assert "c" not in set(filtered["code"])
    assert len(targets) <= 3
    assert weighted["target_weight"].sum() <= 1.0


def test_portfolio_unknown_industry_uses_independent_limit():
    data = candidates()
    data.loc[0:1, "industry_code"] = "UNKNOWN"
    constraints = PortfolioConstraints(
        target_positions=4,
        hard_max_positions=4,
        max_industry_names=1,
        max_unknown_industry_names=1,
        max_new_entries_per_day=4,
        min_trade_score=0.0,
    )

    targets = construct_portfolio_targets(data, constraints, "p1")

    assert targets[targets["industry_code"] == "UNKNOWN"]["code"].tolist() == ["a"]
    assert len(targets[targets["industry_code"] == "I1"]) <= 1
