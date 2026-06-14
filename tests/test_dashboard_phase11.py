from __future__ import annotations

from pathlib import Path

import pandas as pd

from dashboard.queries import (
    data_health_summary,
    fixed_horizon_compare,
    live_monitor,
    model_bundle_summary,
    run_registry,
    score_mode_compare,
    signal_preview,
    walkforward_compare,
)
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


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
    ]
    for path in expected:
        file_path = Path(path)
        assert file_path.exists(), path
        text = file_path.read_text(encoding="utf-8")
        assert "subprocess" not in text
        assert "run_ml_" not in text
        assert "run_live_pipeline" not in text


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
