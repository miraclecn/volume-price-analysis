from __future__ import annotations

import json

import duckdb
import pandas as pd

from ml_stock_selector.runtime.artifacts import prepare_run_artifact_dir, write_backtest_fold_artifacts


def test_run_and_backtest_artifacts_capture_config_params_and_outputs(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[model]\nname = 'test'\n", encoding="utf-8")

    run_root = prepare_run_artifact_dir(
        tmp_path / "runs",
        "run_a",
        config_path=config_path,
        run_manifest={"run_type": "backtest", "score_version": "v2_three_model"},
    )
    backtest_root = write_backtest_fold_artifacts(
        run_root,
        fold_id="wf_2025",
        strategy_id="holding_aware_v2",
        score_version="v2_three_model",
        portfolio_id="wf_2025_holding_aware_v2",
        backtest_params={"initial_cash": 1000.0, "execution": {"slippage_bps": 5.0}},
        targets=pd.DataFrame([{"trade_date": "2025-01-02", "code": "000001.SZ"}]),
        diagnostics=pd.DataFrame([{"trade_date": "2025-01-02", "final_selected_count": 1}]),
        orders=pd.DataFrame([{"sim_date": "2025-01-03", "code": "000001.SZ", "order_seq": 1}]),
        positions=pd.DataFrame([{"sim_date": "2025-01-03", "code": "000001.SZ"}]),
        nav=pd.DataFrame([{"sim_date": "2025-01-03", "nav": 1001.0}]),
        metrics=pd.DataFrame([{"metric_name": "annualized_return", "metric_value": 0.1}]),
    )

    run_manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    params = json.loads((backtest_root / "backtest_params.json").read_text(encoding="utf-8"))
    metrics = json.loads((backtest_root / "metrics.json").read_text(encoding="utf-8"))
    nav = duckdb.connect(":memory:").execute(
        "select nav from read_parquet(?)",
        [str(backtest_root / "nav.parquet")],
    ).fetchone()

    assert run_manifest["run_id"] == "run_a"
    assert (run_root / "config_snapshot.toml").exists()
    assert (run_root / "config_hash.txt").exists()
    assert params["execution"]["slippage_bps"] == 5.0
    assert metrics[0]["metric_name"] == "annualized_return"
    assert nav == (1001.0,)
