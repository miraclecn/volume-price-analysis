from __future__ import annotations

import pandas as pd

from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.constructor import (
    construct_portfolio_targets_v2,
    get_portfolio_diagnostics,
)
from ml_stock_selector.portfolio.holding_policy import HoldingPolicy


def _constraints(**overrides) -> PortfolioConstraints:
    values = {
        "target_positions": 5,
        "hard_max_positions": 5,
        "max_initial_entries": 5,
        "max_new_entries_per_day": 2,
        "max_industry_names": 99,
        "max_unknown_industry_names": 99,
        "min_adv20_amount": 0.0,
        "candidate_min_trade_score": 0.65,
        "core_min_trade_score": 0.75,
        "candidate_absolute_min_rank_pct": 0.0,
        "candidate_active_min_rank_pct": 0.0,
        "candidate_risk_max_rank_pct": 1.0,
        "core_absolute_min_rank_pct": 0.0,
        "core_active_min_rank_pct": 0.0,
        "core_risk_max_rank_pct": 1.0,
        "exclude_bse": True,
        "holding_policy": HoldingPolicy(),
    }
    values.update(overrides)
    return PortfolioConstraints(**values)


def _candidates(codes: list[str], *, trade_date: str = "2024-01-05") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "code": code,
                "industry_code": f"I{idx}",
                "industry_name": f"Industry {idx}",
                "trade_score_v2": 0.95 - idx * 0.01,
                "absolute_rank_pct": 0.90,
                "active_rank_pct": 0.90,
                "risk_rank_pct": 0.20,
                "risk_prob": 0.10,
                "is_st": False,
                "is_paused": False,
                "is_bse": False,
                "can_buy_next_open": True,
                "can_sell_next_open": True,
                "adv20_amount": 100_000_000.0,
            }
            for idx, code in enumerate(codes)
        ]
    )


def _holdings(codes: list[str], *, holding_days: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": code,
                "entry_date": "2024-01-02",
                "entry_price": 10.0 + idx,
                "shares": 100.0,
                "holding_days": holding_days,
                "calendar_days": holding_days,
                "entry_trade_score": 0.80,
                "latest_trade_score": 0.80,
                "entry_reason": "core_pool",
                "industry_code": f"H{idx}",
                "industry_name": f"Held {idx}",
            }
            for idx, code in enumerate(codes)
        ]
    )


def test_retains_holding_before_min_days_even_when_absent_from_candidate_pool():
    targets = construct_portfolio_targets_v2(
        _candidates(["n0", "n1", "n2"]),
        _constraints(),
        "p1",
        current_holdings=_holdings(["h0"], holding_days=1),
    )

    held = targets[targets["code"] == "h0"].iloc[0]
    assert held["entry_reason"] == "retained_holding"
    assert held["hold_reason"] == "hold_due_to_min_days"
    assert "h0" in set(targets["code"])


def test_existing_holdings_do_not_count_against_new_entry_limit():
    targets = construct_portfolio_targets_v2(
        _candidates(["h0", "h1", "n0", "n1", "n2", "n3"]),
        _constraints(target_positions=4, hard_max_positions=4, max_new_entries_per_day=1),
        "p1",
        current_holdings=_holdings(["h0", "h1"], holding_days=1),
    )

    selected = set(targets["code"])
    assert {"h0", "h1"}.issubset(selected)
    assert len(selected - {"h0", "h1"}) == 1


def test_sell_blocked_holding_is_retained_with_blocked_reason():
    data = _candidates(["h0", "n0"])
    data.loc[data["code"] == "h0", "trade_score_v2"] = 0.10
    data.loc[data["code"] == "h0", "can_sell_next_open"] = False

    targets = construct_portfolio_targets_v2(
        data,
        _constraints(),
        "p1",
        current_holdings=_holdings(["h0"], holding_days=5),
    )

    held = targets[targets["code"] == "h0"].iloc[0]
    diagnostics = get_portfolio_diagnostics(targets).iloc[0]
    assert held["entry_reason"] == "sell_blocked"
    assert held["exit_reason"] == "score_exit"
    assert held["sell_blocked_reason"] == "score_exit"
    assert diagnostics["sell_blocked_count"] == 1
