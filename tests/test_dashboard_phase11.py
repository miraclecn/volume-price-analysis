from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from dashboard.queries import (
    backtest_nav,
    continuous_nav,
    continuous_variant_options,
    data_health_summary,
    fixed_horizon_compare,
    fold_metric_matrix,
    live_monitor,
    live_sim_accounts,
    live_sim_nav,
    live_sim_order_summary,
    model_bundle_summary,
    run_dimensions,
    run_metadata,
    run_registry,
    score_mode_compare,
    selection_options,
    signal_preview,
    walkforward_compare,
)
from ml_stock_selector.storage import init_ml_db, upsert_dataframe
from ml_stock_selector.serving.live_sim import init_live_sim_db


def test_dashboard_phase11_pages_exist_and_are_read_only():
    expected = [
        "dashboard/app.py",
        "dashboard/pages/1_Run_Registry.py",
        "dashboard/pages/2_Walkforward_Compare.py",
        "dashboard/pages/3_Score_Mode_Compare.py",
        "dashboard/pages/4_Fixed_Horizon_Compare.py",
        "dashboard/pages/5_Fold_Detail.py",
        "dashboard/pages/6_Model_Bundle.py",
        "dashboard/pages/7_Portfolio_Diagnostics.py",
        "dashboard/pages/8_Signal_Preview.py",
        "dashboard/pages/9_Live_Monitor.py",
        "dashboard/pages/10_Data_Health.py",
        "dashboard/pages/11_Continuous_Curve.py",
        "dashboard/pages/12_Live_Sim.py",
    ]
    for path in expected:
        file_path = Path(path)
        assert file_path.exists(), path
        text = file_path.read_text(encoding="utf-8")
        assert "subprocess" not in text
        assert "run_ml_" not in text
        assert "run_live_pipeline" not in text


def test_dashboard_entrypoints_bootstrap_repo_root_for_streamlit():
    entrypoints = [
        Path("dashboard/app.py"),
        *sorted(Path("dashboard/pages").glob("*.py")),
    ]
    for file_path in entrypoints:
        text = file_path.read_text(encoding="utf-8")
        assert "sys.path.insert(0, str(Path(__file__).resolve().parents[" in text


def test_dashboard_queries_cover_phase8_to_phase10_tables(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    upsert_dataframe(
        con,
        "ml_runs",
        pd.DataFrame(
            [
                {
                    "run_id": "run",
                    "run_type": "walkforward",
                    "experiment_name": "expanding_gap",
                    "status": "success",
                    "feature_set_id": "vpa_d_sequence",
                    "label_version": "from_next_open_h5",
                    "score_version": "v2_absolute_risk_filter",
                    "config_hash": "hash",
                    "git_commit": "abc",
                    "created_at": "t",
                }
            ]
        ),
        ["run_id"],
    )
    metrics = pd.DataFrame(
        [
            {"run_id": "run", "fold_id": "wf_2024", "strategy_id": "holding_aware_v2", "score_version": "v2_absolute_risk_filter", "metric_name": "annual_return", "metric_value": 0.4, "segment": "fold"},
            {"run_id": "run", "fold_id": "wf_2024", "strategy_id": "holding_aware_v2", "score_version": "v2_absolute_risk_filter", "metric_name": "max_drawdown", "metric_value": -0.2, "segment": "fold"},
            {"run_id": "run", "fold_id": "all", "strategy_id": "holding_aware_v2", "score_version": "v2_absolute_risk_filter", "metric_name": "mean_annual_return", "metric_value": 0.4, "segment": "walkforward"},
            {"run_id": "run", "fold_id": "all", "strategy_id": None, "score_version": "comparison", "metric_name": "risk_exit_benefit", "metric_value": -0.03, "segment": "comparison"},
        ]
    )
    upsert_dataframe(con, "ml_backtest_metrics", metrics, ["run_id", "fold_id", "score_version", "metric_name", "segment"])
    upsert_dataframe(
        con,
        "ml_model_bundles",
        pd.DataFrame(
            [
                {
                    "bundle_id": "core_bundle",
                    "run_id": "run",
                    "bundle_role": "production",
                    "absolute_model_id": "abs",
                    "active_model_id": "act",
                    "risk_model_id": "risk",
                    "feature_set_id": "vpa_d_sequence",
                    "label_base": "from_next_open",
                    "horizon_d": 5,
                    "score_version": "v2_absolute_risk_filter",
                    "artifact_dir": "artifacts/core",
                    "status": "active",
                    "created_at": "t",
                }
            ]
        ),
        ["bundle_id"],
    )
    upsert_dataframe(
        con,
        "live_target_positions",
        pd.DataFrame(
            [
                {
                    "trade_date": "2026-06-12",
                    "account_id": "paper",
                    "strategy_id": "holding_aware_v2",
                    "code": "000001.SZ",
                    "target_weight": 0.05,
                    "target_value": 50000.0,
                    "source_bundle_id": "core_bundle",
                    "source_sleeve": "core",
                    "score_version": "v2_absolute_risk_filter",
                    "reason": "core_pool",
                    "generated_at": "t",
                }
            ]
        ),
        ["trade_date", "account_id", "strategy_id", "code"],
    )
    upsert_dataframe(
        con,
        "live_orders",
        pd.DataFrame(
            [
                {
                    "order_id": "ord",
                    "trade_date": "2026-06-13",
                    "account_id": "paper",
                    "strategy_id": "holding_aware_v2",
                    "code": "000001.SZ",
                    "side": "buy",
                    "order_qty": 1000.0,
                    "order_price": 10.0,
                    "status": "created",
                    "created_at": "t",
                }
            ]
        ),
        ["order_id"],
    )

    assert run_registry(con).iloc[0]["annual_return_mean"] == 0.4
    assert walkforward_compare(con).iloc[0]["metric_name"] == "mean_annual_return"
    assert score_mode_compare(con).iloc[0]["score_version"] == "v2_absolute_risk_filter"
    assert fixed_horizon_compare(con).iloc[0]["metric_name"] == "risk_exit_benefit"
    assert model_bundle_summary(con).iloc[0]["bundle_id"] == "core_bundle"
    assert signal_preview(con).iloc[0]["source_sleeve"] == "core"
    assert live_monitor(con).iloc[0]["order_count"] == 1
    assert "table_count" in data_health_summary(con)
    con.close()


def test_dashboard_run_centric_queries_filter_by_selected_dimensions(tmp_path):
    con = init_ml_db(tmp_path / "ml.duckdb")
    upsert_dataframe(
        con,
        "ml_runs",
        pd.DataFrame(
            [
                {
                    "run_id": "run_a",
                    "run_type": "walkforward",
                    "experiment_name": "expanding_gap",
                    "status": "success",
                    "feature_set_id": "vpa_d_sequence",
                    "label_version": "from_next_open_h5",
                    "score_version": "v2_three_model",
                    "config_hash": "hash_a",
                    "git_commit": "abc",
                    "artifact_root": "outputs/ml/runs/run_a",
                    "created_at": "2026-06-13T00:00:00",
                },
                {
                    "run_id": "run_b",
                    "run_type": "walkforward",
                    "experiment_name": "rolling_gap",
                    "status": "success",
                    "feature_set_id": "vpa_d_sequence",
                    "label_version": "from_next_open_h5",
                    "score_version": "v2_absolute_only",
                    "config_hash": "hash_b",
                    "git_commit": "def",
                    "artifact_root": "outputs/ml/runs/run_b",
                    "created_at": "2026-06-14T00:00:00",
                },
            ]
        ),
        ["run_id"],
    )
    upsert_dataframe(
        con,
        "ml_run_folds",
        pd.DataFrame(
            [
                {"run_id": "run_a", "fold_id": "wf_2024", "train_start": "2018-01-01", "train_end": "2022-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31", "gap_type": "one_year_gap", "status": "success"},
                {"run_id": "run_b", "fold_id": "wf_2025", "train_start": "2019-01-01", "train_end": "2023-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31", "gap_type": "one_year_gap", "status": "success"},
            ]
        ),
        ["run_id", "fold_id"],
    )
    upsert_dataframe(
        con,
        "ml_backtest_nav",
        pd.DataFrame(
            [
                {"run_id": "run_a", "fold_id": "wf_2024", "strategy_id": "holding_aware_v2", "score_version": "v2_three_model", "sim_date": "2024-01-02", "nav": 1.00, "cash": 0.1, "gross_exposure": 0.9, "turnover": 0.0},
                {"run_id": "run_a", "fold_id": "wf_2024", "strategy_id": "holding_aware_v2", "score_version": "v2_three_model", "sim_date": "2024-01-03", "nav": 1.08, "cash": 0.2, "gross_exposure": 0.8, "turnover": 0.1},
                {"run_id": "run_b", "fold_id": "wf_2025", "strategy_id": "holding_aware_v2", "score_version": "v2_absolute_only", "sim_date": "2025-01-02", "nav": 0.98, "cash": 0.3, "gross_exposure": 0.7, "turnover": 0.2},
            ]
        ),
        ["run_id", "fold_id", "strategy_id", "score_version", "sim_date"],
    )
    upsert_dataframe(
        con,
        "ml_backtest_metrics",
        pd.DataFrame(
            [
                {"run_id": "run_a", "fold_id": "wf_2024", "strategy_id": "holding_aware_v2", "score_version": "v2_three_model", "metric_name": "annual_return", "metric_value": 0.4, "segment": "fold"},
                {"run_id": "run_a", "fold_id": "wf_2024", "strategy_id": "holding_aware_v2", "score_version": "v2_three_model", "metric_name": "max_drawdown", "metric_value": -0.2, "segment": "fold"},
                {"run_id": "run_b", "fold_id": "wf_2025", "strategy_id": "holding_aware_v2", "score_version": "v2_absolute_only", "metric_name": "annual_return", "metric_value": -0.1, "segment": "fold"},
            ]
        ),
        ["run_id", "fold_id", "score_version", "metric_name", "segment"],
    )

    dimensions = run_dimensions(con)
    assert dimensions["run_id"].tolist() == ["run_b", "run_a"]
    assert run_metadata(con, "run_a").iloc[0]["config_hash"] == "hash_a"
    nav = backtest_nav(con, run_id="run_a", fold_id="wf_2024", strategy_id="holding_aware_v2", score_version="v2_three_model")
    assert nav["nav"].tolist() == [1.0, 1.08]
    assert fold_metric_matrix(con, run_id="run_a").iloc[0]["annual_return"] == 0.4
    con.close()


def test_dashboard_run_centric_queries_fallback_to_legacy_backtest_tables():
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            """
            create table ml_backtest_nav (
                run_id varchar,
                fold_id varchar,
                sim_date varchar,
                nav double,
                cash double,
                gross_exposure double,
                turnover double
            )
            """
        )
        con.execute(
            """
            create table ml_backtest_metrics (
                run_id varchar,
                fold_id varchar,
                score_version varchar,
                metric_name varchar,
                metric_value double,
                segment varchar
            )
            """
        )
        con.execute(
            """
            insert into ml_backtest_nav values
            ('legacy_run', 'wf_2024', '2024-01-02', 1.0, 0.1, 0.9, 0.0),
            ('legacy_run', 'wf_2024', '2024-01-03', 1.1, 0.2, 0.8, 0.1)
            """
        )
        con.execute(
            """
            insert into ml_backtest_metrics values
            ('legacy_run', 'wf_2024', 'v2_three_model', 'annualized_return', 0.25, 'fold'),
            ('legacy_run', 'wf_2024', 'v2_three_model', 'max_drawdown', -0.12, 'fold')
            """
        )

        assert run_dimensions(con).iloc[0]["run_id"] == "legacy_run"
        assert selection_options(con, "legacy_run").iloc[0]["fold_id"] == "wf_2024"
        assert backtest_nav(con, run_id="legacy_run").shape[0] == 2
        metrics = fold_metric_matrix(con, run_id="legacy_run")
        assert metrics.iloc[0]["annual_return"] == 0.25
        assert metrics.iloc[0]["max_drawdown"] == -0.12
    finally:
        con.close()


def test_dashboard_continuous_nav_stitches_yearly_fold_returns_without_reset():
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            """
            create table ml_backtest_nav (
                run_id varchar,
                fold_id varchar,
                strategy_id varchar,
                score_version varchar,
                sim_date varchar,
                nav double,
                cash double,
                gross_exposure double,
                turnover double
            )
            """
        )
        con.execute(
            """
            insert into ml_backtest_nav values
            ('run', 'wf_2020', 'holding_aware_v2', 'v2_three_model', '2020-01-02', 100.0, 0.0, 1.0, 0.0),
            ('run', 'wf_2020', 'holding_aware_v2', 'v2_three_model', '2020-01-03', 110.0, 0.0, 1.0, 0.0),
            ('run', 'wf_2021', 'holding_aware_v2', 'v2_three_model', '2021-01-04', 100.0, 0.0, 1.0, 0.0),
            ('run', 'wf_2021', 'holding_aware_v2', 'v2_three_model', '2021-01-05', 120.0, 0.0, 1.0, 0.0)
            """
        )

        stitched = continuous_nav(
            con,
            run_id="run",
            start_year=2020,
            end_year=2021,
            strategy_id="holding_aware_v2",
            score_version="v2_three_model",
        )

        assert stitched["fold_id"].tolist() == ["wf_2020", "wf_2020", "wf_2021", "wf_2021"]
        assert stitched["continuous_nav"].round(6).tolist() == [100.0, 110.0, 110.0, 132.0]
        assert stitched["fold_return"].round(6).tolist() == [0.0, 0.1, 0.0, 0.2]
    finally:
        con.close()


def test_dashboard_continuous_nav_uses_one_named_strategy_variant_across_years():
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            """
            create table ml_backtest_nav (
                run_id varchar,
                fold_id varchar,
                strategy_id varchar,
                score_version varchar,
                sim_date varchar,
                nav double,
                cash double,
                gross_exposure double,
                turnover double
            )
            """
        )
        con.execute(
            """
            create table ml_backtest_metrics (
                run_id varchar,
                fold_id varchar,
                strategy_id varchar,
                score_version varchar,
                metric_name varchar,
                metric_value double,
                segment varchar
            )
            """
        )
        con.execute(
            """
            insert into ml_backtest_nav values
            ('run', 'wf_2020', 'base', 'v1', '2020-01-02', 100.0, 0.0, 1.0, 0.0),
            ('run', 'wf_2020', 'base', 'v1', '2020-01-03', 105.0, 0.0, 1.0, 0.0),
            ('run', 'wf_2021', 'base', 'v1', '2021-01-04', 100.0, 0.0, 1.0, 0.0),
            ('run', 'wf_2021', 'base', 'v1', '2021-01-05', 106.0, 0.0, 1.0, 0.0),
            ('run', 'wf_2020_absolute_risk_filter_score_adv_combo_top12', 'wf_2020_absolute_risk_filter_score_adv_combo_top12', 'v2_abs_risk_lowadv_full015_top12', '2020-01-02', 100.0, 0.0, 1.0, 0.0),
            ('run', 'wf_2020_absolute_risk_filter_score_adv_combo_top12', 'wf_2020_absolute_risk_filter_score_adv_combo_top12', 'v2_abs_risk_lowadv_full015_top12', '2020-01-03', 131.0, 0.0, 1.0, 0.0),
            ('run', 'wf_2021_absolute_risk_filter_score_adv_combo_top12', 'wf_2021_absolute_risk_filter_score_adv_combo_top12', 'v2_abs_risk_lowadv_full015_top12', '2021-01-04', 100.0, 0.0, 1.0, 0.0),
            ('run', 'wf_2021_absolute_risk_filter_score_adv_combo_top12', 'wf_2021_absolute_risk_filter_score_adv_combo_top12', 'v2_abs_risk_lowadv_full015_top12', '2021-01-05', 130.0, 0.0, 1.0, 0.0)
            """
        )
        con.execute(
            """
            insert into ml_backtest_metrics values
            ('run', 'wf_2020_absolute_risk_filter_score_adv_combo_top12', 'wf_2020_absolute_risk_filter_score_adv_combo_top12', 'v2_abs_risk_lowadv_full015_top12', 'annual_return', 0.31, 'fold'),
            ('run', 'wf_2021_absolute_risk_filter_score_adv_combo_top12', 'wf_2021_absolute_risk_filter_score_adv_combo_top12', 'v2_abs_risk_lowadv_full015_top12', 'annual_return', 0.30, 'fold')
            """
        )

        options = continuous_variant_options(con, run_id="run", start_year=2020, end_year=2021)
        assert options.iloc[0]["fold_suffix"] == "absolute_risk_filter_score_adv_combo_top12"
        assert options.iloc[0]["min_annual_return"] == 0.30

        stitched = continuous_nav(
            con,
            run_id="run",
            start_year=2020,
            end_year=2021,
            fold_suffix="absolute_risk_filter_score_adv_combo_top12",
            score_version="v2_abs_risk_lowadv_full015_top12",
        )

        assert stitched["fold_id"].tolist() == [
            "wf_2020_absolute_risk_filter_score_adv_combo_top12",
            "wf_2020_absolute_risk_filter_score_adv_combo_top12",
            "wf_2021_absolute_risk_filter_score_adv_combo_top12",
            "wf_2021_absolute_risk_filter_score_adv_combo_top12",
        ]
        assert stitched["continuous_nav"].round(6).tolist() == [100.0, 131.0, 131.0, 170.3]
    finally:
        con.close()


def test_dashboard_live_sim_queries_return_nav_drawdown_and_order_summary(tmp_path):
    con = init_live_sim_db(tmp_path / "live_sim.duckdb")
    try:
        con.execute("insert into live_sim_account values ('paper', 300000, 'now')")
        con.execute(
            """
            insert into live_sim_nav values
            ('paper', 'INITIAL', 300000, 300000, 0, 0, 0),
            ('paper', '2026-06-05', 300000, 300000, 0, 0, 0),
            ('paper', '2026-06-06', 330000, 30000, 300000, 0.10, 0),
            ('paper', '2026-06-07', 297000, 20000, 277000, -0.01, -0.10)
            """
        )
        con.execute(
            """
            insert into live_sim_planned_orders
            (account_id, decision_date, execution_date, code, side, target_weight, trade_score_v2,
             absolute_rank_pct, active_rank_pct, risk_rank_pct, adv20_amount, estimated_price,
             estimated_qty, target_value, entry_reason, signal_action, status, generated_at)
            values ('paper', '2026-06-05', '2026-06-06', '000001.SZ', 'buy', 0.5, 0.9,
                    0.9, 0.8, 0.2, 20000000, 10, 1000, 150000, 'core_pool', 'buy', 'planned', 'now')
            """
        )
        con.execute(
            """
            insert into live_sim_executions
            (account_id, decision_date, sim_date, code, side, qty, target_weight, fill_px,
             commission, stamp_duty, fees, status, reason, realized_pnl, generated_at)
            values ('paper', '2026-06-05', '2026-06-06', '000001.SZ', 'buy', 1000, 0.5, 10,
                    3, 0, 3, 'filled', null, null, 'now')
            """
        )

        assert live_sim_accounts(con)["account_id"].tolist() == ["paper"]
        nav = live_sim_nav(con, "paper")
        assert nav["daily_return"].round(6).tolist() == [0.0, 0.0, 0.1, -0.1]
        assert nav["drawdown"].round(6).tolist() == [0.0, 0.0, 0.0, -0.1]
        summary = live_sim_order_summary(con, "paper")
        assert summary["planned_count"].sum() == 1
        assert summary["filled_count"].sum() == 1
    finally:
        con.close()


def test_dashboard_main_is_run_centric_visual_workspace():
    text = Path("dashboard/app.py").read_text(encoding="utf-8")
    assert "render_run_dashboard" in text
    assert "Continuous Curve" in text
    assert "continuous_nav" in text
    assert "selectbox" in text
    assert "line_chart" in text
    assert "bar_chart" in text


def test_dashboard_has_dedicated_continuous_curve_page():
    text = Path("dashboard/pages/11_Continuous_Curve.py").read_text(encoding="utf-8")
    assert "Continuous Curve" in text
    assert "continuous_nav" in text
    assert "Start Year" in text
    assert "End Year" in text


def test_dashboard_has_dedicated_live_sim_page():
    text = Path("dashboard/pages/12_Live_Sim.py").read_text(encoding="utf-8")
    assert "Live Sim" in text
    assert "live_sim_nav" in text
    assert "line_chart" in text


def test_dashboard_queries_tolerate_uninitialized_database():
    con = duckdb.connect(":memory:")
    try:
        assert run_dimensions(con).empty
        assert run_metadata(con, "missing").empty
        assert backtest_nav(con, run_id="missing").empty
        assert fold_metric_matrix(con, run_id="missing").empty
        assert run_registry(con).empty
        assert walkforward_compare(con).empty
        assert score_mode_compare(con).empty
        assert fixed_horizon_compare(con).empty
        assert model_bundle_summary(con).empty
        assert signal_preview(con).empty
        assert live_monitor(con).empty
        assert data_health_summary(con)["table_count"] == 0
    finally:
        con.close()
