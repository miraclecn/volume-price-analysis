from __future__ import annotations

import pandas as pd


TRADEABILITY_COLUMNS = [
    "industry_code",
    "industry_name",
    "is_st",
    "is_paused",
    "is_bse",
    "adv20_amount",
    "can_buy_next_open",
    "can_sell_next_open",
    "next_open",
    "next_limit_up",
    "next_limit_down",
    "next_is_paused",
]


def load_backtest_candidates(
    con,
    *,
    run_id: str | None = None,
    fold_id: str | None = None,
    score_version: str | None = None,
    exclude_bse: bool = True,
) -> pd.DataFrame:
    where = []
    params: list[object] = []
    if run_id is not None:
        where.append("p.run_id = ?")
        params.append(run_id)
    if fold_id is not None:
        where.append("p.fold_id = ?")
        params.append(fold_id)
    if score_version is not None:
        where.append("p.score_version = ?")
        params.append(score_version)
    sql = f"""
        select
            p.*,
            t.industry_code,
            t.industry_name,
            t.is_st,
            t.is_paused,
            t.is_bse,
            t.adv20_amount,
            t.can_buy_next_open,
            t.can_sell_next_open,
            t.next_open,
            t.next_limit_up,
            t.next_limit_down,
            t.next_is_paused
        from ml_predictions_daily p
        left join ml_tradeability_daily t
          on p.trade_date = t.trade_date and p.code = t.code
        {"where " + " and ".join(where) if where else ""}
        order by p.trade_date, p.code
    """
    candidates = con.execute(sql, params).fetchdf()
    if exclude_bse and "is_bse" in candidates:
        candidates = candidates[~candidates["is_bse"].fillna(False).astype(bool)].copy()
    return candidates.reset_index(drop=True)
