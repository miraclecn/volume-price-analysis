from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from ml_stock_selector.backtest.execution import ExecutionConfig
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.holding_policy import HoldingPolicy
from ml_stock_selector.serving.live_sim import (
    LiveSimConfig,
    archived_adv_score,
    generate_markdown_report,
    init_live_sim_db,
    run_live_sim_day,
)


def test_archived_adv_score_matches_preferred_formula() -> None:
    predictions = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "absolute_rank_pct": 0.90, "adv20_amount": 10.0},
            {"trade_date": "2024-01-02", "code": "b", "absolute_rank_pct": 0.80, "adv20_amount": 20.0},
            {"trade_date": "2024-01-02", "code": "c", "absolute_rank_pct": 0.70, "adv20_amount": 30.0},
        ]
    )

    scored = archived_adv_score(predictions)

    assert scored.loc[scored["code"] == "a", "full_prediction_pool_adv_pct"].item() == pytest.approx(1 / 3)
    assert scored.loc[scored["code"] == "a", "trade_score_v2"].item() == pytest.approx(0.85 * 0.90 + 0.15 * (1 - 1 / 3))
    assert scored.loc[scored["code"] == "a", "alpha_rank_pct"].item() == pytest.approx(0.90)
    assert scored.loc[scored["code"] == "a", "active_rank_pct"].item() == pytest.approx(0.90)
    assert scored.loc[scored["code"] == "a", "core_score"].item() == pytest.approx(scored.loc[scored["code"] == "a", "trade_score_v2"].item())
    assert scored.loc[scored["code"] == "a", "trade_score"].item() == pytest.approx(scored.loc[scored["code"] == "a", "trade_score_v2"].item())
    assert scored["score_version"].eq("preferred_adv10m_fulladv015_top12").all()


def test_live_sim_default_constraints_match_archived_preferred_replay() -> None:
    constraints = LiveSimConfig().constraints
    holding_policy = constraints.holding_policy

    assert constraints.target_positions == 12
    assert constraints.hard_max_positions == 15
    assert constraints.max_initial_entries == 12
    assert constraints.max_new_entries_per_day == 4
    assert constraints.min_adv20_amount == 10_000_000
    assert constraints.candidate_min_trade_score == pytest.approx(0.75)
    assert constraints.core_min_trade_score == pytest.approx(0.75)
    assert constraints.candidate_absolute_min_rank_pct == pytest.approx(0.70)
    assert constraints.candidate_active_min_rank_pct == pytest.approx(0.70)
    assert constraints.candidate_risk_max_rank_pct == pytest.approx(0.65)
    assert constraints.core_absolute_min_rank_pct == pytest.approx(0.75)
    assert constraints.core_active_min_rank_pct == pytest.approx(0.65)
    assert constraints.core_risk_max_rank_pct == pytest.approx(0.55)
    assert constraints.exclude_bse is True
    assert holding_policy.min_hold_days == 3
    assert holding_policy.target_hold_days == 5
    assert holding_policy.max_hold_days == 10
    assert holding_policy.sell_score_threshold == pytest.approx(0.45)
    assert holding_policy.risk_exit_rank_pct == pytest.approx(0.85)
    assert holding_policy.risk_exit_prob == pytest.approx(0.70)
    assert holding_policy.sell_if_not_candidate_after_target_days is True
    assert holding_policy.force_exit_after_max_hold_days is True
    assert holding_policy.allow_score_exit_before_min_hold is False


def test_run_live_sim_day_initializes_cash_plans_next_day_and_settles_without_duplicates(tmp_path: Path) -> None:
    state_db = init_live_sim_db(tmp_path / "live.duckdb")
    predictions = _prediction_frame("2024-01-02")
    bars = _bars_frame()
    config = LiveSimConfig(
        account_id="paper",
        initial_cash=300_000.0,
        portfolio_id="preferred",
        target_positions=2,
        report_dir=tmp_path / "reports",
        execution=ExecutionConfig(allow_fractional_shares=False),
        constraints=PortfolioConstraints(
            target_positions=2,
            hard_max_positions=2,
            max_initial_entries=2,
            max_new_entries_per_day=2,
            min_adv20_amount=10_000_000.0,
            candidate_min_trade_score=0.0,
            core_min_trade_score=0.0,
            candidate_absolute_min_rank_pct=0.0,
            candidate_active_min_rank_pct=0.0,
            candidate_risk_max_rank_pct=0.65,
            core_absolute_min_rank_pct=0.0,
            core_active_min_rank_pct=0.0,
            core_risk_max_rank_pct=0.65,
            holding_policy=HoldingPolicy(sell_score_threshold=-1.0),
        ),
    )

    first = run_live_sim_day(state_db, "2024-01-02", predictions, bars, config)
    second = run_live_sim_day(state_db, "2024-01-02", predictions, bars, config)

    assert first.plan_date == "2024-01-02"
    assert first.execution_date == "2024-01-03"
    assert first.nav["cash"] == pytest.approx(300_000.0)
    assert len(first.planned_orders) == 2
    assert first.executions.empty
    assert len(second.planned_orders) == 2
    assert state_db.execute("select count(*) from live_sim_planned_orders").fetchone()[0] == 2
    assert state_db.execute("select count(*) from live_sim_nav where sim_date = '2024-01-02'").fetchone()[0] == 1

    settled = run_live_sim_day(state_db, "2024-01-03", _prediction_frame("2024-01-03"), bars, config)

    assert len(settled.executions[settled.executions["status"] == "filled"]) == 2
    assert state_db.execute("select count(*) from live_sim_executions where status = 'filled'").fetchone()[0] == 2
    assert state_db.execute("select count(*) from live_sim_holdings where qty > 0").fetchone()[0] == 2
    assert settled.nav["cash"] < 300_000.0
    assert settled.nav["nav"] == pytest.approx(settled.nav["cash"] + settled.nav["holding_market_value"])

    repeated_settle = run_live_sim_day(state_db, "2024-01-03", _prediction_frame("2024-01-03"), bars, config)

    assert state_db.execute("select count(*) from live_sim_executions where status = 'filled'").fetchone()[0] == 2
    assert repeated_settle.executions.empty
    state_db.close()


def test_generate_markdown_report_includes_account_trade_holdings_and_risk(tmp_path: Path) -> None:
    con = init_live_sim_db(tmp_path / "live.duckdb")
    result = run_live_sim_day(
        con,
        "2024-01-02",
        _prediction_frame("2024-01-02"),
        _bars_frame(),
        LiveSimConfig(
            account_id="paper",
            initial_cash=300_000.0,
            portfolio_id="preferred",
            target_positions=2,
            report_dir=tmp_path / "reports",
            execution=ExecutionConfig(allow_fractional_shares=False),
            constraints=PortfolioConstraints(
                target_positions=2,
                hard_max_positions=2,
                max_initial_entries=2,
                max_new_entries_per_day=2,
                min_adv20_amount=10_000_000.0,
                candidate_min_trade_score=0.0,
                core_min_trade_score=0.0,
                candidate_absolute_min_rank_pct=0.0,
                candidate_active_min_rank_pct=0.0,
                candidate_risk_max_rank_pct=0.65,
                core_absolute_min_rank_pct=0.0,
                core_active_min_rank_pct=0.0,
                core_risk_max_rank_pct=0.65,
                holding_policy=HoldingPolicy(sell_score_threshold=-1.0),
            ),
        ),
    )

    report = generate_markdown_report(result)

    assert "账户摘要" in report
    assert "当日成交" in report
    assert "当前持仓" in report
    assert "下一交易日计划" in report
    assert "最大回撤" in report
    assert "300,000.00" in report
    con.close()


def test_settlement_keeps_existing_holdings_not_in_due_plan(tmp_path: Path) -> None:
    con = init_live_sim_db(tmp_path / "live.duckdb")
    config = LiveSimConfig(
        account_id="paper",
        initial_cash=300_000.0,
        portfolio_id="preferred",
        target_positions=2,
        report_dir=tmp_path / "reports",
        execution=ExecutionConfig(allow_fractional_shares=False),
        constraints=PortfolioConstraints(
            target_positions=2,
            hard_max_positions=2,
            max_initial_entries=2,
            max_new_entries_per_day=2,
            min_adv20_amount=10_000_000.0,
            candidate_min_trade_score=0.0,
            core_min_trade_score=0.0,
            candidate_absolute_min_rank_pct=0.0,
            candidate_active_min_rank_pct=0.0,
            candidate_risk_max_rank_pct=0.65,
            core_absolute_min_rank_pct=0.0,
            core_active_min_rank_pct=0.0,
            core_risk_max_rank_pct=0.65,
            holding_policy=HoldingPolicy(sell_score_threshold=-1.0),
        ),
    )
    con.execute("insert into live_sim_account values ('paper', 300000, 'now')")
    con.execute("insert into live_sim_nav values ('paper', '2024-01-02', 300000, 290000, 10000, 0, 0)")
    con.execute(
        "insert into live_sim_holdings values ('paper', '000001.SZ', 1000, '2024-01-02', 10, 0.8, 'core_pool', 'now')"
    )
    con.execute(
        """
        insert into live_sim_planned_orders
        (account_id, decision_date, execution_date, code, side, target_weight, trade_score_v2,
         absolute_rank_pct, active_rank_pct, risk_rank_pct, adv20_amount, estimated_price,
         estimated_qty, target_value, entry_reason, signal_action, status, generated_at)
        values ('paper', '2024-01-02', '2024-01-03', '000002.SZ', 'buy', 0.5, 0.9,
                0.9, 0.8, 0.2, 20000000, 10, 1000, 150000, 'core_pool', 'buy', 'planned', 'now')
        """
    )

    result = run_live_sim_day(con, "2024-01-03", _prediction_frame("2024-01-03"), _bars_frame(), config)

    assert not ((result.executions["code"] == "000001.SZ") & (result.executions["side"] == "sell")).any()
    assert con.execute("select qty from live_sim_holdings where account_id = 'paper' and code = '000001.SZ'").fetchone()[0] == 1000
    con.close()


def _prediction_frame(trade_date: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "code": "000001.SZ",
                "industry_code": "I1",
                "industry_name": "Industry 1",
                "is_st": False,
                "is_paused": False,
                "is_bse": False,
                "adv20_amount": 12_000_000.0,
                "absolute_rank_pct": 0.95,
                "active_rank_pct": 0.80,
                "risk_rank_pct": 0.20,
            },
            {
                "trade_date": trade_date,
                "code": "000002.SZ",
                "industry_code": "I2",
                "industry_name": "Industry 2",
                "is_st": False,
                "is_paused": False,
                "is_bse": False,
                "adv20_amount": 20_000_000.0,
                "absolute_rank_pct": 0.90,
                "active_rank_pct": 0.75,
                "risk_rank_pct": 0.25,
            },
            {
                "trade_date": trade_date,
                "code": "000003.SZ",
                "industry_code": "I3",
                "industry_name": "Industry 3",
                "is_st": False,
                "is_paused": False,
                "is_bse": False,
                "adv20_amount": 8_000_000.0,
                "absolute_rank_pct": 0.99,
                "active_rank_pct": 0.90,
                "risk_rank_pct": 0.10,
            },
        ]
    )


def _bars_frame() -> pd.DataFrame:
    rows = []
    for trade_date, open_px, close_px in [
        ("2024-01-02", 10.0, 10.0),
        ("2024-01-03", 10.2, 10.4),
        ("2024-01-04", 10.5, 10.6),
    ]:
        for code in ["000001.SZ", "000002.SZ", "000003.SZ"]:
            rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "open": open_px,
                    "close": close_px,
                    "limit_up": open_px + 1.0,
                    "limit_down": open_px - 1.0,
                    "is_paused": False,
                }
            )
    return pd.DataFrame(rows)
