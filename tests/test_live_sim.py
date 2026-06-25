from __future__ import annotations

import json
import hashlib
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from ml_stock_selector.backtest.execution import ExecutionConfig
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.holding_policy import HoldingPolicy
from ml_stock_selector.serving.live_sim import (
    PROFIT_PROTECT_PORTFOLIO_ID,
    PROFIT_PROTECT_RUN_ID,
    PROFIT_PROTECT_SCORE_VERSION,
    SCORE_VERSION,
    LiveSimConfig,
    activate_profit_protect_live_bundle,
    archived_adv_score,
    generate_markdown_report,
    init_live_sim_db,
    load_active_live_model_bundle,
    load_live_predictions,
    profit_protect_live_sim_config,
    run_live_sim_day,
    save_live_strategy_config_snapshot,
    live_sim_reproducibility_snapshot,
    upsert_live_predictions,
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


def test_live_sim_reproducibility_snapshot_records_active_model_and_backtest_parameters() -> None:
    snapshot = live_sim_reproducibility_snapshot(LiveSimConfig())

    assert snapshot["score_version"] == SCORE_VERSION
    assert snapshot["portfolio_id"] == "preferred_adv10m_fulladv015_top12"
    assert snapshot["initial_cash"] == 300_000.0
    assert snapshot["execution"] == {
        "slippage_bps": 5.0,
        "commission_bps": 3.0,
        "stamp_duty_bps": 5.0,
        "allow_fractional_shares": False,
        "a_share_lot_size": 100,
        "execution_price": "next_open",
    }
    assert snapshot["constraints"]["target_positions"] == 12
    assert snapshot["constraints"]["min_adv20_amount"] == 10_000_000.0
    assert snapshot["constraints"]["holding_policy"]["target_hold_days"] == 5


def test_profit_protect_live_config_matches_archived_main_strategy(tmp_path: Path) -> None:
    config = profit_protect_live_sim_config(report_dir=tmp_path / "reports")
    holding_policy = config.constraints.holding_policy

    assert config.account_id == "profit_protect_paper"
    assert config.initial_cash == 1_000_000.0
    assert config.portfolio_id == PROFIT_PROTECT_PORTFOLIO_ID
    assert config.score_version == PROFIT_PROTECT_SCORE_VERSION
    assert config.execution.slippage_bps == pytest.approx(10.0)
    assert config.execution.commission_bps == pytest.approx(3.0)
    assert config.execution.stamp_duty_bps == pytest.approx(5.0)
    assert config.execution.allow_fractional_shares is False
    assert config.constraints.candidate_risk_max_rank_pct == pytest.approx(0.55)
    assert config.constraints.core_risk_max_rank_pct == pytest.approx(0.45)
    assert holding_policy.sell_score_threshold == pytest.approx(0.35)
    assert holding_policy.risk_exit_rank_pct == pytest.approx(0.75)
    assert holding_policy.risk_exit_prob == pytest.approx(0.60)
    assert config.profit_protect_enabled is True
    assert config.profit_protect_min_days == 3
    assert config.profit_protect_min_gain == pytest.approx(0.03)
    assert config.profit_protect_exit_below == pytest.approx(0.005)
    assert config.market_zero_below == pytest.approx(0.375)
    assert config.market_half_below == pytest.approx(0.475)


def test_live_sim_db_owns_runtime_bundle_config_and_recent_prediction_tables(tmp_path: Path) -> None:
    con = init_live_sim_db(tmp_path / "live.duckdb")

    tables = {
        row[0]
        for row in con.execute(
            "select table_name from information_schema.tables where table_schema = 'main'"
        ).fetchall()
    }

    assert "live_model_bundle" in tables
    assert "live_strategy_config_snapshot" in tables
    assert "live_predictions_daily" in tables
    con.close()


def test_activate_profit_protect_live_bundle_records_reproducible_artifacts(tmp_path: Path) -> None:
    con = init_live_sim_db(tmp_path / "live.duckdb")

    bundle = activate_profit_protect_live_bundle(con)
    active = load_active_live_model_bundle(con, PROFIT_PROTECT_PORTFOLIO_ID)

    assert bundle["bundle_id"] == active["bundle_id"]
    assert active["strategy_id"] == PROFIT_PROTECT_PORTFOLIO_ID
    assert active["score_version"] == PROFIT_PROTECT_SCORE_VERSION
    assert active["source_run_id"] == PROFIT_PROTECT_RUN_ID
    assert Path(str(active["alpha_artifact_uri"])).exists()
    assert Path(str(active["risk_artifact_uri"])).exists()
    assert Path(str(active["feature_schema_uri"])).exists()
    assert active["alpha_rounds"] == 160
    assert active["risk_rounds"] == 120
    assert active["train_window_mode"] == "fixed_round_full_history"
    assert active["source_manifest_hash"]
    con.close()


def test_activate_profit_protect_live_bundle_can_snapshot_small_live_artifacts(tmp_path: Path) -> None:
    manifest_root = tmp_path / "manifests"
    fold_dir = manifest_root / f"run_id={PROFIT_PROTECT_RUN_ID}" / "fold_id=wf_2026"
    artifact_dir = tmp_path / "research_artifacts"
    snapshot_dir = tmp_path / "live" / "artifacts" / "profit"
    fold_dir.mkdir(parents=True)
    artifact_dir.mkdir()

    alpha_pkl = artifact_dir / "alpha.pkl"
    risk_pkl = artifact_dir / "risk.pkl"
    schema = fold_dir / "feature_schema.json"
    alpha_pkl.write_bytes(b"alpha")
    risk_pkl.write_bytes(b"risk")
    alpha_pkl.with_suffix(".params.json").write_text('{"rounds": 160}', encoding="utf-8")
    risk_pkl.with_suffix(".params.json").write_text('{"rounds": 120}', encoding="utf-8")
    schema.write_text('{"features": []}', encoding="utf-8")
    (artifact_dir / "train_matrix.npy").write_bytes(b"large-cache")
    manifest = {
        "fold_id": "wf_2026",
        "feature_set_id": "vpa_d_sequence_fundamental_v1",
        "label_base": "from_next_open",
        "horizon_d": 5,
        "train_window_mode": "fixed_round_full_history",
        "source_train_window_mode": "expanding_no_gap",
        "fixed_alpha_rounds": 160,
        "fixed_risk_rounds": 120,
        "artifacts": {
            "absolute": {
                "model_id": "alpha",
                "artifact_uri": str(alpha_pkl),
                "feature_schema_uri": str(schema),
                "feature_set_id": "vpa_d_sequence_fundamental_v1",
                "label_base": "from_next_open",
                "horizon_d": 5,
            },
            "risk": {
                "model_id": "risk",
                "artifact_uri": str(risk_pkl),
                "feature_schema_uri": str(schema),
                "feature_set_id": "vpa_d_sequence_fundamental_v1",
                "label_base": "from_next_open",
                "horizon_d": 5,
            },
        },
    }
    manifest_path = fold_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    con = init_live_sim_db(tmp_path / "live.duckdb")

    active = activate_profit_protect_live_bundle(
        con,
        manifest_root=manifest_root,
        artifact_snapshot_dir=snapshot_dir,
    )

    assert Path(str(active["alpha_artifact_uri"])).is_relative_to(snapshot_dir)
    assert Path(str(active["risk_artifact_uri"])).is_relative_to(snapshot_dir)
    assert Path(str(active["feature_schema_uri"])).is_relative_to(snapshot_dir)
    assert Path(str(active["source_manifest_path"])).is_relative_to(snapshot_dir)
    assert Path(str(active["alpha_artifact_uri"])).read_bytes() == b"alpha"
    assert (snapshot_dir / "alpha.params.json").exists()
    assert (snapshot_dir / "risk.params.json").exists()
    assert (snapshot_dir / "manifest.json").exists()
    assert (snapshot_dir / "feature_schema.json").exists()
    assert not (snapshot_dir / "train_matrix.npy").exists()
    assert active["source_manifest_hash"] == expected_hash
    con.close()


def test_live_predictions_are_stored_in_live_db_without_copying_shared_feature_tables(tmp_path: Path) -> None:
    con = init_live_sim_db(tmp_path / "live.duckdb")
    bundle = activate_profit_protect_live_bundle(con)
    predictions = pd.DataFrame(
        [
            {
                "trade_date": "2026-06-16",
                "code": "000001.SZ",
                "model_id": "alpha_risk:a:r",
                "horizon_d": 5,
                "absolute_score": 0.1,
                "absolute_rank_pct": 0.2,
                "risk_prob": 0.3,
                "risk_rank_pct": 0.4,
                "trade_score_v2": 0.5,
                "adv20_amount": 20_000_000.0,
                "generated_at": "now",
            }
        ]
    )

    upsert_live_predictions(con, predictions, bundle_id=str(bundle["bundle_id"]))
    loaded = load_live_predictions(con, "2026-06-16", bundle_id=str(bundle["bundle_id"]))
    tables = {
        row[0]
        for row in con.execute(
            "select table_name from information_schema.tables where table_schema = 'main'"
        ).fetchall()
    }

    assert loaded["code"].tolist() == ["000001.SZ"]
    assert loaded.iloc[0]["strategy_id"] == PROFIT_PROTECT_PORTFOLIO_ID
    assert loaded.iloc[0]["score_version"] == PROFIT_PROTECT_SCORE_VERSION
    assert loaded.iloc[0]["absolute_rank_pct"] == pytest.approx(0.2)
    assert "ml_feature_mart_daily" not in tables
    assert "ml_tradeability_daily" not in tables
    con.close()


def test_live_strategy_config_snapshot_preserves_active_runtime_parameters(tmp_path: Path) -> None:
    con = init_live_sim_db(tmp_path / "live.duckdb")
    config = profit_protect_live_sim_config()

    snapshot = save_live_strategy_config_snapshot(con, config)
    stored = con.execute(
        """
        select strategy_id, account_id, score_version, config_json
        from live_strategy_config_snapshot
        where snapshot_id = ?
        """,
        [snapshot["snapshot_id"]],
    ).fetchone()

    assert stored[0] == PROFIT_PROTECT_PORTFOLIO_ID
    assert stored[1] == "profit_protect_paper"
    assert stored[2] == PROFIT_PROTECT_SCORE_VERSION
    assert '"profit_protect_enabled": true' in stored[3]
    assert '"risk_exit_rank_pct": 0.75' in stored[3]
    con.close()


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


def test_profit_protect_plans_exit_after_gain_reversal(tmp_path: Path) -> None:
    con = init_live_sim_db(tmp_path / "live.duckdb")
    config = LiveSimConfig(
        account_id="paper",
        initial_cash=300_000.0,
        portfolio_id="profit_protect",
        target_positions=1,
        report_dir=tmp_path / "reports",
        profit_protect_enabled=True,
        profit_protect_min_days=3,
        profit_protect_min_gain=0.03,
        profit_protect_exit_below=0.005,
        constraints=PortfolioConstraints(
            target_positions=1,
            hard_max_positions=2,
            max_initial_entries=1,
            max_new_entries_per_day=1,
            min_adv20_amount=0.0,
            candidate_min_trade_score=0.0,
            core_min_trade_score=0.0,
            candidate_absolute_min_rank_pct=0.0,
            candidate_active_min_rank_pct=0.0,
            candidate_risk_max_rank_pct=1.0,
            core_absolute_min_rank_pct=0.0,
            core_active_min_rank_pct=0.0,
            core_risk_max_rank_pct=1.0,
            holding_policy=HoldingPolicy(
                min_hold_days=3,
                target_hold_days=5,
                max_hold_days=10,
                sell_score_threshold=-1.0,
                sell_if_not_candidate_after_target_days=True,
            ),
        ),
    )
    con.execute("insert into live_sim_account values ('paper', 300000, 'now')")
    con.executemany(
        "insert into live_sim_nav values ('paper', ?, 300000, 299000, 1000, 0, 0)",
        [(date,) for date in ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]],
    )
    con.execute(
        "insert into live_sim_holdings values ('paper', '000001.SZ', 100, '2024-01-02', 10, 0.8, 'core_pool', 'now')"
    )
    predictions = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-08",
                "code": "000001.SZ",
                "industry_code": "I1",
                "industry_name": "Industry 1",
                "is_st": False,
                "is_paused": False,
                "is_bse": False,
                "adv20_amount": 20_000_000.0,
                "absolute_rank_pct": 0.95,
                "active_rank_pct": 0.90,
                "risk_rank_pct": 0.10,
            }
        ]
    )
    bars = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-08",
                "code": "000001.SZ",
                "open": 10.0,
                "high": 10.35,
                "low": 10.01,
                "close": 10.04,
                "prev_close": 10.10,
                "limit_up": 11.0,
                "limit_down": 9.0,
                "is_paused": False,
            },
            {
                "trade_date": "2024-01-09",
                "code": "000001.SZ",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "prev_close": 10.04,
                "limit_up": 11.0,
                "limit_down": 9.0,
                "is_paused": False,
            },
        ]
    )

    result = run_live_sim_day(con, "2024-01-08", predictions, bars, config)

    planned_sell = result.planned_orders[
        (result.planned_orders["code"] == "000001.SZ")
        & (result.planned_orders["side"] == "sell")
    ]
    assert len(planned_sell) == 1
    assert planned_sell.iloc[0]["exit_reason"] == "profit_protect_exit"
    assert con.execute(
        "select max_high_ret from live_sim_holding_path_stats where account_id = 'paper' and code = '000001.SZ'"
    ).fetchone()[0] == pytest.approx(0.035)
    con.close()


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


def test_live_sim_plans_sell_for_aged_holding_no_longer_in_candidate_pool(tmp_path: Path) -> None:
    con = init_live_sim_db(tmp_path / "live.duckdb")
    config = LiveSimConfig(
        account_id="paper",
        initial_cash=300_000.0,
        portfolio_id="preferred",
        target_positions=1,
        report_dir=tmp_path / "reports",
        execution=ExecutionConfig(allow_fractional_shares=False),
        constraints=PortfolioConstraints(
            target_positions=1,
            hard_max_positions=2,
            max_initial_entries=1,
            max_new_entries_per_day=1,
            min_adv20_amount=0.0,
            candidate_min_trade_score=0.0,
            core_min_trade_score=0.0,
            candidate_absolute_min_rank_pct=0.0,
            candidate_active_min_rank_pct=0.0,
            candidate_risk_max_rank_pct=1.0,
            core_absolute_min_rank_pct=0.0,
            core_active_min_rank_pct=0.0,
            core_risk_max_rank_pct=1.0,
            holding_policy=HoldingPolicy(
                min_hold_days=3,
                target_hold_days=4,
                max_hold_days=10,
                sell_score_threshold=-1.0,
                sell_if_not_candidate_after_target_days=True,
            ),
        ),
    )
    con.execute("insert into live_sim_account values ('paper', 300000, 'now')")
    con.executemany(
        "insert into live_sim_nav values ('paper', ?, 300000, 299000, 1000, 0, 0)",
        [(date,) for date in ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]],
    )
    con.execute(
        "insert into live_sim_holdings values ('paper', '000001.SZ', 100, '2024-01-02', 10, 0.8, 'core_pool', 'now')"
    )
    predictions = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-08",
                "code": "000002.SZ",
                "industry_code": "I2",
                "industry_name": "Industry 2",
                "is_st": False,
                "is_paused": False,
                "is_bse": False,
                "adv20_amount": 20_000_000.0,
                "absolute_rank_pct": 0.95,
                "active_rank_pct": 0.90,
                "risk_rank_pct": 0.10,
            }
        ]
    )
    bars = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "code": code,
                "open": 10.0,
                "close": 10.0,
                "limit_up": 11.0,
                "limit_down": 9.0,
                "is_paused": False,
            }
            for trade_date in ["2024-01-08", "2024-01-09"]
            for code in ["000001.SZ", "000002.SZ"]
        ]
    )

    result = run_live_sim_day(con, "2024-01-08", predictions, bars, config)

    planned_sell = result.planned_orders[
        (result.planned_orders["code"] == "000001.SZ")
        & (result.planned_orders["side"] == "sell")
    ]
    assert len(planned_sell) == 1
    assert planned_sell.iloc[0]["signal_action"] == "sell"
    assert planned_sell.iloc[0]["estimated_qty"] == 100
    con.close()


def test_live_sim_holding_days_use_observed_trading_dates_not_weekdays(tmp_path: Path) -> None:
    con = init_live_sim_db(tmp_path / "live.duckdb")
    config = LiveSimConfig(
        account_id="paper",
        initial_cash=300_000.0,
        portfolio_id="preferred",
        target_positions=1,
        report_dir=tmp_path / "reports",
        execution=ExecutionConfig(allow_fractional_shares=False),
        constraints=PortfolioConstraints(
            target_positions=1,
            hard_max_positions=2,
            max_initial_entries=1,
            max_new_entries_per_day=1,
            min_adv20_amount=0.0,
            candidate_min_trade_score=0.0,
            core_min_trade_score=0.0,
            candidate_absolute_min_rank_pct=0.0,
            candidate_active_min_rank_pct=0.0,
            candidate_risk_max_rank_pct=1.0,
            core_absolute_min_rank_pct=0.0,
            core_active_min_rank_pct=0.0,
            core_risk_max_rank_pct=1.0,
            holding_policy=HoldingPolicy(
                min_hold_days=0,
                target_hold_days=4,
                max_hold_days=10,
                sell_score_threshold=-1.0,
                sell_if_not_candidate_after_target_days=True,
            ),
        ),
    )
    con.execute("insert into live_sim_account values ('paper', 300000, 'now')")
    con.executemany(
        "insert into live_sim_nav values ('paper', ?, 300000, 299000, 1000, 0, 0)",
        [(date,) for date in ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"]],
    )
    con.execute(
        "insert into live_sim_holdings values ('paper', '000001.SZ', 100, '2024-01-02', 10, 0.8, 'core_pool', 'now')"
    )
    predictions = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-08",
                "code": "000002.SZ",
                "industry_code": "I2",
                "industry_name": "Industry 2",
                "is_st": False,
                "is_paused": False,
                "is_bse": False,
                "adv20_amount": 20_000_000.0,
                "absolute_rank_pct": 0.95,
                "active_rank_pct": 0.90,
                "risk_rank_pct": 0.10,
            }
        ]
    )
    bars = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "code": code,
                "open": 10.0,
                "close": 10.0,
                "limit_up": 11.0,
                "limit_down": 9.0,
                "is_paused": False,
            }
            for trade_date in ["2024-01-08", "2024-01-09"]
            for code in ["000001.SZ", "000002.SZ"]
        ]
    )

    result = run_live_sim_day(con, "2024-01-08", predictions, bars, config)

    if not result.planned_orders.empty:
        assert not (
            (result.planned_orders["code"] == "000001.SZ")
            & (result.planned_orders["side"] == "sell")
        ).any()
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
