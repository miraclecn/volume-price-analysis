from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml_stock_selector.constants import UNKNOWN_INDUSTRY_CODE
from ml_stock_selector.portfolio.holding_policy import HoldingPolicy
from ml_stock_selector.universe import detect_is_bse


@dataclass(frozen=True)
class PortfolioConstraints:
    target_positions: int = 12
    hard_max_positions: int = 15
    max_initial_entries: int = 12
    max_industry_names: int = 3
    max_unknown_industry_names: int = 1
    max_new_entries_per_day: int = 4
    min_adv20_amount: float | None = None
    min_trade_score: float = 0.65
    allow_cash: bool = True
    min_candidate_pool_size: int = 5
    candidate_min_count: int | None = None
    candidate_absolute_min_rank_pct: float = 0.70
    candidate_active_min_rank_pct: float = 0.70
    candidate_risk_max_rank_pct: float = 0.65
    core_absolute_min_rank_pct: float = 0.75
    core_active_min_rank_pct: float = 0.65
    core_risk_max_rank_pct: float = 0.55
    core_min_trade_score: float = 0.75
    candidate_min_trade_score: float = 0.65
    exclude_bse: bool = True
    holding_policy: HoldingPolicy = HoldingPolicy()

    def __post_init__(self) -> None:
        if self.candidate_min_count is not None:
            object.__setattr__(self, "min_candidate_pool_size", self.candidate_min_count)
        if self.holding_policy.sell_score_threshold >= self.candidate_min_trade_score:
            raise ValueError("sell_score_threshold must be below candidate_min_trade_score")


def apply_hard_filters(candidates: pd.DataFrame, constraints: PortfolioConstraints, score_column: str | None = "trade_score") -> pd.DataFrame:
    out = candidates.copy()
    mask = pd.Series(True, index=out.index)
    if constraints.exclude_bse:
        if "is_bse" in out:
            mask &= ~out["is_bse"].fillna(False).astype(bool)
        elif "code" in out:
            mask &= ~out["code"].map(detect_is_bse)
    for column in ["is_st", "is_paused"]:
        if column in out:
            mask &= ~out[column].fillna(False).astype(bool)
    if "can_buy_next_open" in out:
        mask &= out["can_buy_next_open"].fillna(False).astype(bool)
    if constraints.min_adv20_amount is not None and "adv20_amount" in out:
        mask &= out["adv20_amount"].fillna(0.0) >= constraints.min_adv20_amount
    if score_column and score_column in out:
        threshold = constraints.candidate_min_trade_score if score_column == "trade_score_v2" else constraints.min_trade_score
        mask &= out[score_column].fillna(-1.0) >= threshold
    return out[mask].copy()


def is_unknown_industry(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip().upper() == UNKNOWN_INDUSTRY_CODE
