from __future__ import annotations

import pandas as pd

from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.constructor import (
    get_portfolio_diagnostics,
    construct_portfolio_targets_v2,
)
from ml_stock_selector.portfolio.holding_policy import HoldingPolicy
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


def test_portfolio_diagnostics_table_schema_and_upsert(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    columns = {
        row[0]
        for row in con.execute(
            """
            select column_name
            from information_schema.columns
            where table_schema = 'main'
              and table_name = 'ml_portfolio_construction_diagnostics'
            """
        ).fetchall()
    }
    diagnostics = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "run_id": "run",
                "fold_id": "wf_2020",
                "portfolio_id": "p1",
                "score_version": "v2_three_model",
                "raw_candidate_count": 10,
                "hard_filter_pass_count": 8,
                "core_pool_size": 2,
                "candidate_pool_size": 6,
                "selected_from_core": 2,
                "selected_from_candidate": 4,
                "final_selected_count": 6,
                "low_adv_rejected_count": 1,
                "cannot_buy_rejected_count": 1,
                "st_rejected_count": 0,
                "paused_rejected_count": 0,
                "bse_rejected_count": 0,
                "low_trade_score_rejected_count": 0,
                "high_risk_rejected_count": 2,
                "industry_limit_blocked_count": 0,
                "unknown_industry_limit_blocked_count": 0,
                "max_new_entries_blocked_count": 0,
                "retained_holdings_count": 1,
                "sell_signal_count": 1,
                "sell_executed_count": 0,
                "sell_blocked_count": 1,
                "hold_due_to_min_days_count": 1,
                "hold_due_to_score_ok_count": 0,
                "exit_due_to_score_count": 1,
                "exit_due_to_risk_count": 0,
                "exit_due_to_time_count": 0,
                "exit_due_to_not_candidate_count": 0,
                "avg_holding_days_current": 2.0,
                "median_holding_days_current": 2.0,
                "cash_weight": 0.4,
                "created_at": "t",
            }
        ]
    )

    upsert_dataframe(
        con,
        "ml_portfolio_construction_diagnostics",
        diagnostics,
        ["trade_date", "run_id", "fold_id", "portfolio_id", "score_version"],
    )
    row = con.execute(
        "select selected_from_candidate from ml_portfolio_construction_diagnostics"
    ).fetchone()
    con.close()

    assert {
        "trade_date",
        "run_id",
        "fold_id",
        "portfolio_id",
        "score_version",
        "raw_candidate_count",
        "hard_filter_pass_count",
        "core_pool_size",
        "candidate_pool_size",
        "selected_from_core",
        "selected_from_candidate",
        "final_selected_count",
        "low_adv_rejected_count",
        "cannot_buy_rejected_count",
        "st_rejected_count",
        "paused_rejected_count",
        "bse_rejected_count",
        "low_trade_score_rejected_count",
        "high_risk_rejected_count",
        "industry_limit_blocked_count",
        "unknown_industry_limit_blocked_count",
        "max_new_entries_blocked_count",
        "retained_holdings_count",
        "sell_signal_count",
        "sell_executed_count",
        "sell_blocked_count",
        "hold_due_to_min_days_count",
        "hold_due_to_score_ok_count",
        "exit_due_to_score_count",
        "exit_due_to_risk_count",
        "exit_due_to_time_count",
        "exit_due_to_not_candidate_count",
        "avg_holding_days_current",
        "median_holding_days_current",
        "cash_weight",
        "created_at",
    }.issubset(columns)
    assert row == (4,)


def test_construct_portfolio_targets_v2_attaches_one_diagnostic_row_per_day():
    day1 = _day("2024-01-02", "a")
    day2 = _day("2024-01-03", "b")
    data = pd.concat([day1, day2], ignore_index=True)

    targets = construct_portfolio_targets_v2(
        data,
            PortfolioConstraints(
                target_positions=2,
                hard_max_positions=2,
                max_initial_entries=2,
                max_new_entries_per_day=2,
            max_industry_names=99,
            min_adv20_amount=0,
            min_candidate_pool_size=1,
            candidate_min_trade_score=0.0,
            core_min_trade_score=0.0,
            holding_policy=HoldingPolicy(sell_score_threshold=-1.0),
        ),
        "p1",
    )
    diagnostics = get_portfolio_diagnostics(targets)

    assert diagnostics["trade_date"].tolist() == ["2024-01-02", "2024-01-03"]
    assert diagnostics["final_selected_count"].tolist() == [2, 2]


def _day(trade_date: str, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "code": f"{prefix}{idx}",
                "industry_code": f"I{idx}",
                "industry_name": f"Industry {idx}",
                "trade_score_v2": 0.9 - idx * 0.01,
                "absolute_rank_pct": 0.8,
                "active_rank_pct": 0.8,
                "risk_rank_pct": 0.2,
                "is_st": False,
                "is_paused": False,
                "can_buy_next_open": True,
                "adv20_amount": 100.0,
                "is_bse": False,
                "run_id": "run",
                "fold_id": "wf_2020",
                "score_version": "v2_three_model",
            }
            for idx in range(3)
        ]
    )
