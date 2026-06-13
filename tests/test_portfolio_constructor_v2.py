from __future__ import annotations

import pandas as pd

from ml_stock_selector.portfolio.constraints import PortfolioConstraints, apply_hard_filters
from ml_stock_selector.portfolio.constructor import (
    get_portfolio_diagnostics,
    construct_portfolio_targets_v2,
)


def _v2_candidates(
    count: int,
    *,
    trade_date: str = "2024-01-02",
    prefix: str = "s",
    core_count: int | None = None,
) -> pd.DataFrame:
    core_count = count if core_count is None else core_count
    rows = []
    for idx in range(count):
        is_core = idx < core_count
        rows.append(
            {
                "trade_date": trade_date,
                "code": f"{prefix}{idx:02d}",
                "industry_code": f"I{idx:02d}",
                "industry_name": f"Industry {idx:02d}",
                "trade_score_v2": 0.95 - idx * 0.01,
                "absolute_rank_pct": 0.82 if is_core else 0.72,
                "active_rank_pct": 0.74 if is_core else 0.71,
                "risk_rank_pct": 0.30 if is_core else 0.60,
                "is_st": False,
                "is_paused": False,
                "can_buy_next_open": True,
                "adv20_amount": 100_000_000.0,
                "is_bse": False,
                "run_id": "run",
                "fold_id": "wf_2020",
                "score_version": "v2_three_model",
            }
        )
    return pd.DataFrame(rows)


def _constraints(**overrides) -> PortfolioConstraints:
    values = {
        "target_positions": 12,
        "hard_max_positions": 15,
        "max_initial_entries": 12,
        "max_new_entries_per_day": 4,
        "max_industry_names": 99,
        "max_unknown_industry_names": 99,
        "min_adv20_amount": 50_000_000,
        "min_trade_score": 0.65,
        "candidate_min_trade_score": 0.65,
        "core_min_trade_score": 0.75,
        "min_candidate_pool_size": 1,
        "exclude_bse": True,
    }
    values.update(overrides)
    return PortfolioConstraints(**values)


def test_v2_initial_build_uses_max_initial_entries_not_daily_new_limit():
    targets = construct_portfolio_targets_v2(_v2_candidates(12), _constraints(), "p1")

    assert len(targets) == 12
    assert len(targets) > 4
    assert len(targets) <= 15


def test_v2_existing_holdings_do_not_count_against_new_entry_limit():
    held = _v2_candidates(10, prefix="h")
    new = _v2_candidates(8, prefix="n")
    new["trade_score_v2"] = [0.90 - idx * 0.01 for idx in range(len(new))]
    data = pd.concat([held, new], ignore_index=True)
    current_holdings = pd.DataFrame({"code": held["code"].tolist()})

    targets = construct_portfolio_targets_v2(
        data,
        _constraints(target_positions=14, max_initial_entries=12, max_new_entries_per_day=4),
        "p1",
        current_holdings=current_holdings,
    )

    selected = set(targets["code"])
    new_selected = selected - set(current_holdings["code"])
    assert len(targets) == 14
    assert set(current_holdings["code"]).issubset(selected)
    assert len(new_selected) <= 4


def test_v2_candidate_pool_fills_when_core_pool_is_short():
    data = _v2_candidates(10, core_count=2)

    targets = construct_portfolio_targets_v2(data, _constraints(target_positions=8), "p1")

    assert len(targets) == 8
    assert targets["entry_reason"].tolist()[:2] == ["core_pool", "core_pool"]
    assert (targets["entry_reason"] == "candidate_pool").sum() == 6


def test_v2_candidate_pool_can_select_when_core_pool_is_empty():
    data = _v2_candidates(8, core_count=0)

    targets = construct_portfolio_targets_v2(data, _constraints(target_positions=6), "p1")

    assert len(targets) == 6
    assert set(targets["entry_reason"]) == {"candidate_pool"}


def test_v2_bucketed_selection_takes_leaders_from_each_score_bucket():
    data = _v2_candidates(40)

    targets = construct_portfolio_targets_v2(
        data,
        _constraints(
            target_positions=12,
            core_min_trade_score=0.451,
            candidate_min_trade_score=0.451,
            selection_bucket_count=4,
            selection_per_bucket=3,
        ),
        "p1",
    )

    assert targets["code"].tolist() == [
        "s00",
        "s01",
        "s02",
        "s10",
        "s11",
        "s12",
        "s20",
        "s21",
        "s22",
        "s30",
        "s31",
        "s32",
    ]


def test_v2_hard_filters_include_bse_st_pause_buy_liquidity_and_trade_score():
    data = _v2_candidates(7)
    data.loc[1, "is_bse"] = True
    data.loc[2, "is_st"] = True
    data.loc[3, "is_paused"] = True
    data.loc[4, "can_buy_next_open"] = False
    data.loc[5, "adv20_amount"] = 1_000_000.0
    data.loc[6, "trade_score_v2"] = 0.10

    filtered = apply_hard_filters(data, _constraints(), score_column="trade_score_v2")
    targets = construct_portfolio_targets_v2(data, _constraints(target_positions=7), "p1")
    diagnostics = get_portfolio_diagnostics(targets)

    assert filtered["code"].tolist() == ["s00"]
    assert targets["code"].tolist() == ["s00"]
    row = diagnostics.iloc[0]
    assert row["raw_candidate_count"] == 7
    assert row["hard_filter_pass_count"] == 1
    assert row["bse_rejected_count"] == 1
    assert row["st_rejected_count"] == 1
    assert row["paused_rejected_count"] == 1
    assert row["cannot_buy_rejected_count"] == 1
    assert row["low_adv_rejected_count"] == 1
    assert row["low_trade_score_rejected_count"] == 1


def test_v2_bse_diagnostics_fall_back_to_code_suffix_when_flag_is_absent():
    data = _v2_candidates(2)
    data.loc[1, "code"] = "430001.BJ"
    data = data.drop(columns=["is_bse"])

    targets = construct_portfolio_targets_v2(data, _constraints(target_positions=2), "p1")
    diagnostics = get_portfolio_diagnostics(targets)

    assert targets["code"].tolist() == ["s00"]
    assert diagnostics.iloc[0]["bse_rejected_count"] == 1
