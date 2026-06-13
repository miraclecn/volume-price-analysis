from __future__ import annotations

import pandas as pd

from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constraints import PortfolioConstraints, apply_hard_filters
from ml_stock_selector.portfolio.constructor import construct_portfolio_targets, construct_portfolio_targets_v2


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


def test_portfolio_constructs_targets_per_trade_date():
    data = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"] * 3 + ["2024-01-03"] * 3,
            "code": ["a", "b", "c", "d", "e", "f"],
            "industry_code": ["I1", "I2", "I3", "I1", "I2", "I3"],
            "trade_score": [0.99, 0.98, 0.10, 0.97, 0.96, 0.20],
            "is_st": [False] * 6,
            "is_paused": [False] * 6,
            "can_buy_next_open": [True] * 6,
            "adv20_amount": [100.0] * 6,
        }
    )
    constraints = PortfolioConstraints(
        target_positions=2,
        hard_max_positions=2,
        max_industry_names=2,
        max_new_entries_per_day=2,
        min_trade_score=0.8,
    )

    targets = construct_portfolio_targets(data, constraints, "p1")

    assert targets.groupby("trade_date")["code"].apply(list).to_dict() == {
        "2024-01-02": ["a", "b"],
        "2024-01-03": ["d", "e"],
    }


def test_allocate_weights_allocates_independently_per_trade_date():
    selected = pd.DataFrame(
        {
            "trade_date": sum(([f"2024-01-{day:02d}"] * 2 for day in range(2, 17)), []),
            "code": [f"code_{idx:02d}" for idx in range(30)],
        }
    )

    weighted = allocate_weights(selected, 0.05, 0.10, allow_cash=True)

    assert weighted.groupby("trade_date")["target_weight"].sum().eq(0.20).all()


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


def test_v2_portfolio_uses_candidate_pool_when_core_pool_missing():
    data = candidates().assign(
        absolute_rank_pct=[0.75, 0.74, 0.73, 0.72, 0.71],
        active_rank_pct=[0.60, 0.60, 0.60, 0.60, 0.60],
        risk_rank_pct=[0.50, 0.50, 0.50, 0.50, 0.50],
        trade_score_v2=[0.82, 0.81, 0.80, 0.79, 0.78],
    )
    constraints = PortfolioConstraints(target_positions=3, hard_max_positions=3, min_trade_score=0.0, candidate_min_count=1)

    targets = construct_portfolio_targets_v2(data, constraints, "p1")

    assert targets["code"].tolist() == ["a", "b", "d"]
    assert set(targets["entry_reason"]) == {"candidate_pool"}


def test_v2_portfolio_uses_core_pool_and_unknown_limit():
    data = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"] * 4,
            "code": ["u1", "u2", "i1", "i2"],
            "industry_code": ["UNKNOWN", "UNKNOWN", "I1", "I2"],
            "trade_score_v2": [0.95, 0.94, 0.93, 0.92],
            "absolute_rank_pct": [0.90, 0.89, 0.88, 0.87],
            "active_rank_pct": [0.90, 0.89, 0.88, 0.87],
            "risk_rank_pct": [0.10, 0.10, 0.10, 0.10],
            "is_st": [False] * 4,
            "is_paused": [False] * 4,
            "can_buy_next_open": [True] * 4,
            "adv20_amount": [100.0] * 4,
        }
    )
    constraints = PortfolioConstraints(
        target_positions=4,
        hard_max_positions=4,
        max_unknown_industry_names=1,
        max_new_entries_per_day=4,
        min_trade_score=0.0,
        candidate_min_count=1,
    )

    targets = construct_portfolio_targets_v2(data, constraints, "p1")

    assert targets[targets["industry_code"] == "UNKNOWN"]["code"].tolist() == ["u1"]
    assert "u2" not in set(targets["code"])
