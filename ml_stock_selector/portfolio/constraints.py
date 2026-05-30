from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PortfolioConstraints:
    target_positions: int = 12
    hard_max_positions: int = 15
    max_industry_names: int = 3
    max_new_entries_per_day: int = 4
    min_adv20_amount: float | None = None
    min_trade_score: float = 0.80
    allow_cash: bool = True


def apply_hard_filters(candidates: pd.DataFrame, constraints: PortfolioConstraints) -> pd.DataFrame:
    out = candidates.copy()
    mask = pd.Series(True, index=out.index)
    for column in ["is_st", "is_paused"]:
        if column in out:
            mask &= ~out[column].fillna(False).astype(bool)
    if "can_buy_next_open" in out:
        mask &= out["can_buy_next_open"].fillna(False).astype(bool)
    if constraints.min_adv20_amount is not None and "adv20_amount" in out:
        mask &= out["adv20_amount"].fillna(0.0) >= constraints.min_adv20_amount
    if "trade_score" in out:
        mask &= out["trade_score"].fillna(-1.0) >= constraints.min_trade_score
    return out[mask].copy()

