from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ml_stock_selector.portfolio.constraints import (
    PortfolioConstraints,
    apply_hard_filters,
    is_unknown_industry,
)


def construct_portfolio_targets(
    scored_candidates: pd.DataFrame,
    constraints: PortfolioConstraints,
    portfolio_id: str,
    current_holdings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    filtered = apply_hard_filters(scored_candidates, constraints)
    selected_rows = []
    industry_counts: dict[object, int] = {}
    unknown_industry_count = 0
    new_entries = 0
    held = set(current_holdings["code"]) if current_holdings is not None and "code" in current_holdings else set()
    for row in filtered.sort_values(["trade_score", "code"], ascending=[False, True]).itertuples(index=False):
        code = getattr(row, "code")
        industry = getattr(row, "industry_code", None)
        if len(selected_rows) >= min(constraints.target_positions, constraints.hard_max_positions):
            break
        industry_unknown = is_unknown_industry(industry)
        if industry_unknown and unknown_industry_count >= constraints.max_unknown_industry_names:
            continue
        if not industry_unknown and industry_counts.get(industry, 0) >= constraints.max_industry_names:
            continue
        if code not in held and new_entries >= constraints.max_new_entries_per_day:
            continue
        if code not in held:
            new_entries += 1
        row_dict = row._asdict()
        if industry_unknown:
            unknown_industry_count += 1
        else:
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        selected_rows.append(row_dict)
    out = pd.DataFrame(selected_rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "portfolio_id",
                "code",
                "industry_code",
                "industry_name",
                "target_weight",
                "rank_n",
                "trade_score",
                "entry_reason",
                "generated_at",
            ]
        )
    out["portfolio_id"] = portfolio_id
    out["rank_n"] = range(1, len(out) + 1)
    out["target_weight"] = 0.0
    out["entry_reason"] = [
        "trade_score; industry_unknown" if is_unknown_industry(value) else "trade_score"
        for value in out.get("industry_code", pd.Series([None] * len(out)))
    ]
    out["generated_at"] = datetime.now(timezone.utc).isoformat()
    columns = [
        "trade_date",
        "portfolio_id",
        "code",
        "industry_code",
        "industry_name",
        "target_weight",
        "rank_n",
        "trade_score",
        "entry_reason",
        "generated_at",
    ]
    return out[[column for column in columns if column in out.columns]]
