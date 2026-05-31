from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml_stock_selector.constants import UNKNOWN_INDUSTRY_CODE


@dataclass(frozen=True)
class PortfolioConstraints:
    target_positions: int = 12
    hard_max_positions: int = 15
    max_industry_names: int = 3
    max_unknown_industry_names: int = 1
    max_new_entries_per_day: int = 4
    min_adv20_amount: float | None = None
    min_trade_score: float = 0.80
    allow_cash: bool = True
    candidate_min_count: int = 5
    candidate_absolute_min_rank_pct: float = 0.70
    candidate_active_min_rank_pct: float = 0.70
    candidate_risk_max_rank_pct: float = 0.60
    core_absolute_min_rank_pct: float = 0.80
    core_active_min_rank_pct: float = 0.75
    core_risk_max_rank_pct: float = 0.35
    core_min_trade_score: float = 0.80


def apply_hard_filters(candidates: pd.DataFrame, constraints: PortfolioConstraints, score_column: str | None = "trade_score") -> pd.DataFrame:
    out = candidates.copy()
    mask = pd.Series(True, index=out.index)
    for column in ["is_st", "is_paused"]:
        if column in out:
            mask &= ~out[column].fillna(False).astype(bool)
    if "can_buy_next_open" in out:
        mask &= out["can_buy_next_open"].fillna(False).astype(bool)
    if constraints.min_adv20_amount is not None and "adv20_amount" in out:
        mask &= out["adv20_amount"].fillna(0.0) >= constraints.min_adv20_amount
    if score_column and score_column in out:
        mask &= out[score_column].fillna(-1.0) >= constraints.min_trade_score
    return out[mask].copy()


def is_unknown_industry(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip().upper() == UNKNOWN_INDUSTRY_CODE
