from __future__ import annotations

import pandas as pd

from ml_stock_selector.portfolio.holding_policy import (
    HoldingPolicy,
    HoldingState,
    evaluate_sell_decision,
)


def _holding(**overrides) -> HoldingState:
    values = {
        "code": "s00",
        "entry_date": "2024-01-02",
        "entry_price": 10.0,
        "shares": 100.0,
        "holding_days": 1,
        "calendar_days": 1,
        "entry_trade_score": 0.80,
        "latest_trade_score": 0.80,
        "entry_reason": "core_pool",
    }
    values.update(overrides)
    return HoldingState(**values)


def _latest(**overrides) -> pd.Series:
    values = {
        "code": "s00",
        "trade_score_v2": 0.80,
        "risk_rank_pct": 0.20,
        "risk_prob": 0.10,
        "is_bse": False,
        "is_st": False,
        "data_quality_high_severity": False,
        "can_sell_next_open": True,
        "in_candidate_pool": False,
    }
    values.update(overrides)
    return pd.Series(values)


def test_score_drop_before_min_hold_does_not_sell():
    decision = evaluate_sell_decision(
        _holding(holding_days=1),
        _latest(trade_score_v2=0.10, in_candidate_pool=False),
        HoldingPolicy(min_hold_days=3, sell_score_threshold=0.45),
    )

    assert decision.should_sell is False
    assert decision.reason == "hold_due_to_min_days"


def test_risk_exit_can_break_min_hold():
    decision = evaluate_sell_decision(
        _holding(holding_days=1),
        _latest(risk_rank_pct=0.90),
        HoldingPolicy(min_hold_days=3, risk_exit_rank_pct=0.85),
    )

    assert decision.should_sell is True
    assert decision.reason == "risk_exit"
    assert decision.blocked is False


def test_target_hold_allows_exit_when_no_longer_candidate():
    decision = evaluate_sell_decision(
        _holding(holding_days=5),
        _latest(in_candidate_pool=False),
        HoldingPolicy(target_hold_days=5),
    )

    assert decision.should_sell is True
    assert decision.reason == "not_candidate_after_target_days"


def test_max_hold_forces_time_exit():
    decision = evaluate_sell_decision(
        _holding(holding_days=10),
        _latest(in_candidate_pool=True),
        HoldingPolicy(max_hold_days=10),
    )

    assert decision.should_sell is True
    assert decision.reason == "time_exit"


def test_can_sell_false_blocks_exit():
    decision = evaluate_sell_decision(
        _holding(holding_days=5),
        _latest(trade_score_v2=0.10, can_sell_next_open=False),
        HoldingPolicy(min_hold_days=3, sell_score_threshold=0.45),
    )

    assert decision.should_sell is True
    assert decision.reason == "score_exit"
    assert decision.blocked is True


def test_missing_boolean_tradeability_fields_use_defaults():
    decision = evaluate_sell_decision(
        _holding(holding_days=5),
        _latest(
            is_bse=pd.NA,
            is_st=pd.NA,
            data_quality_high_severity=pd.NA,
            can_sell_next_open=pd.NA,
            in_candidate_pool=pd.NA,
            trade_score_v2=0.80,
        ),
        HoldingPolicy(target_hold_days=5),
    )

    assert decision.should_sell is True
    assert decision.reason == "not_candidate_after_target_days"
    assert decision.blocked is False
