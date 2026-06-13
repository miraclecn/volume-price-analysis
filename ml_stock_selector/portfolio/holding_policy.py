from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class HoldingPolicy:
    min_hold_days: int = 3
    target_hold_days: int = 5
    max_hold_days: int = 10
    sell_score_threshold: float = 0.45
    risk_exit_rank_pct: float = 0.85
    risk_exit_prob: float = 0.70
    sell_if_not_candidate_after_target_days: bool = True
    force_exit_after_max_hold_days: bool = True
    allow_score_exit_before_min_hold: bool = False


@dataclass(frozen=True)
class HoldingState:
    code: str
    entry_date: str
    entry_price: float
    shares: float
    holding_days: int
    calendar_days: int
    entry_trade_score: float | None
    latest_trade_score: float | None
    entry_reason: str | None


@dataclass(frozen=True)
class SellDecision:
    code: str
    should_sell: bool
    reason: str
    blocked: bool = False


def holding_policy_from_dict(raw: dict[str, object] | None) -> HoldingPolicy:
    if not raw:
        return HoldingPolicy()
    return HoldingPolicy(
        min_hold_days=int(raw.get("min_hold_days", HoldingPolicy.min_hold_days)),
        target_hold_days=int(raw.get("target_hold_days", HoldingPolicy.target_hold_days)),
        max_hold_days=int(raw.get("max_hold_days", HoldingPolicy.max_hold_days)),
        sell_score_threshold=float(raw.get("sell_score_threshold", HoldingPolicy.sell_score_threshold)),
        risk_exit_rank_pct=float(raw.get("risk_exit_rank_pct", HoldingPolicy.risk_exit_rank_pct)),
        risk_exit_prob=float(raw.get("risk_exit_prob", HoldingPolicy.risk_exit_prob)),
        sell_if_not_candidate_after_target_days=bool(
            raw.get(
                "sell_if_not_candidate_after_target_days",
                HoldingPolicy.sell_if_not_candidate_after_target_days,
            )
        ),
        force_exit_after_max_hold_days=bool(
            raw.get("force_exit_after_max_hold_days", HoldingPolicy.force_exit_after_max_hold_days)
        ),
        allow_score_exit_before_min_hold=bool(
            raw.get("allow_score_exit_before_min_hold", HoldingPolicy.allow_score_exit_before_min_hold)
        ),
    )


def holding_state_from_row(row: pd.Series) -> HoldingState:
    return HoldingState(
        code=str(row.get("code")),
        entry_date=str(row.get("entry_date", row.get("trade_date", ""))),
        entry_price=float(row.get("entry_price", row.get("close", 0.0)) or 0.0),
        shares=float(row.get("shares", row.get("position_qty", row.get("qty", 0.0))) or 0.0),
        holding_days=int(row.get("holding_days", 0) or 0),
        calendar_days=int(row.get("calendar_days", row.get("holding_days", 0)) or 0),
        entry_trade_score=_optional_float(row.get("entry_trade_score", row.get("trade_score_v2"))),
        latest_trade_score=_optional_float(row.get("latest_trade_score", row.get("trade_score_v2"))),
        entry_reason=_optional_str(row.get("entry_reason")),
    )


def evaluate_sell_decision(
    holding: HoldingState,
    latest_row: pd.Series,
    holding_policy: HoldingPolicy,
) -> SellDecision:
    can_sell = _optional_bool(latest_row.get("can_sell_next_open"), True)

    hard_reasons = []
    if _optional_bool(latest_row.get("is_bse"), False):
        hard_reasons.append("is_bse")
    if _optional_bool(latest_row.get("is_st"), False):
        hard_reasons.append("is_st")
    if _optional_bool(latest_row.get("data_quality_high_severity"), False):
        hard_reasons.append("data_quality_high_severity")
    if hard_reasons:
        return _sell_decision(holding.code, f"hard_exit:{','.join(hard_reasons)}", can_sell)

    risk_rank_pct = _optional_float(latest_row.get("risk_rank_pct"))
    risk_prob = _optional_float(latest_row.get("risk_prob"))
    if (
        risk_rank_pct is not None
        and risk_rank_pct >= holding_policy.risk_exit_rank_pct
    ) or (
        risk_prob is not None
        and risk_prob >= holding_policy.risk_exit_prob
    ):
        return _sell_decision(holding.code, "risk_exit", can_sell)

    latest_trade_score = _optional_float(latest_row.get("trade_score_v2", holding.latest_trade_score))
    score_exit_allowed = holding.holding_days >= holding_policy.min_hold_days or holding_policy.allow_score_exit_before_min_hold
    if (
        score_exit_allowed
        and latest_trade_score is not None
        and latest_trade_score < holding_policy.sell_score_threshold
    ):
        return _sell_decision(holding.code, "score_exit", can_sell)

    in_candidate_pool = _optional_bool(latest_row.get("in_candidate_pool"), False)
    if (
        holding.holding_days >= holding_policy.target_hold_days
        and holding_policy.sell_if_not_candidate_after_target_days
        and not in_candidate_pool
    ):
        return _sell_decision(holding.code, "not_candidate_after_target_days", can_sell)

    if (
        holding.holding_days >= holding_policy.max_hold_days
        and holding_policy.force_exit_after_max_hold_days
    ):
        return _sell_decision(holding.code, "time_exit", can_sell)

    if holding.holding_days < holding_policy.min_hold_days and not in_candidate_pool:
        return SellDecision(code=holding.code, should_sell=False, reason="hold_due_to_min_days")
    if latest_trade_score is not None and latest_trade_score >= holding_policy.sell_score_threshold:
        return SellDecision(code=holding.code, should_sell=False, reason="hold_due_to_score_ok")
    return SellDecision(code=holding.code, should_sell=False, reason="hold")


def _sell_decision(code: str, reason: str, can_sell: bool) -> SellDecision:
    return SellDecision(code=code, should_sell=True, reason=reason, blocked=not can_sell)


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_bool(value: object, default: bool) -> bool:
    if value is None or pd.isna(value):
        return default
    return bool(value)


def _optional_str(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)
