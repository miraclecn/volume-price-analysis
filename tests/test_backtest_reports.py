from __future__ import annotations

import pandas as pd

from ml_stock_selector.backtest.reports import (
    portfolio_diagnostics_report_metrics,
    selected_count_distribution,
    write_portfolio_diagnostics_report,
)


def test_portfolio_diagnostics_report_metrics_explain_sparse_targets():
    diagnostics = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "raw_candidate_count": [100, 80, 60],
            "hard_filter_pass_count": [50, 40, 30],
            "core_pool_size": [2, 0, 1],
            "candidate_pool_size": [20, 10, 5],
            "selected_from_core": [2, 0, 1],
            "selected_from_candidate": [6, 8, 0],
            "final_selected_count": [8, 8, 1],
            "low_adv_rejected_count": [10, 8, 6],
            "cannot_buy_rejected_count": [5, 4, 3],
            "st_rejected_count": [1, 0, 0],
            "max_new_entries_blocked_count": [0, 2, 0],
        }
    )

    metrics = portfolio_diagnostics_report_metrics(diagnostics)
    distribution = selected_count_distribution(diagnostics)

    assert metrics["avg_raw_candidate_count"] == 80.0
    assert metrics["avg_hard_filter_pass_count"] == 40.0
    assert metrics["avg_core_pool_size"] == 1.0
    assert metrics["avg_candidate_pool_size"] == 35 / 3
    assert metrics["avg_selected_from_core"] == 1.0
    assert metrics["avg_selected_from_candidate"] == 14 / 3
    assert metrics["low_adv_rejected_count"] == 24.0
    assert metrics["cannot_buy_rejected_count"] == 12.0
    assert metrics["st_rejected_count"] == 1.0
    assert metrics["max_new_entries_blocked_count"] == 2.0
    assert metrics["empty_day_ratio"] == 0.0
    assert metrics["avg_selected_count"] == 17 / 3
    assert distribution.to_dict("records") == [
        {"final_selected_count": 1, "day_count": 1},
        {"final_selected_count": 8, "day_count": 2},
    ]


def test_write_portfolio_diagnostics_report_outputs_metrics_and_distribution(tmp_path):
    diagnostics = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03"],
            "raw_candidate_count": [10, 20],
            "hard_filter_pass_count": [8, 16],
            "core_pool_size": [2, 4],
            "candidate_pool_size": [6, 8],
            "selected_from_core": [2, 3],
            "selected_from_candidate": [4, 5],
            "final_selected_count": [6, 8],
            "low_adv_rejected_count": [1, 2],
            "cannot_buy_rejected_count": [0, 1],
            "st_rejected_count": [0, 0],
            "max_new_entries_blocked_count": [0, 2],
        }
    )

    paths = write_portfolio_diagnostics_report(diagnostics, tmp_path, prefix="wf_2020")

    metrics = pd.read_csv(paths["metrics"])
    distribution = pd.read_csv(paths["selected_count_distribution"])
    assert set(metrics["metric_name"]).issuperset({"avg_raw_candidate_count", "avg_selected_count"})
    assert distribution.to_dict("records") == [
        {"final_selected_count": 6, "day_count": 1},
        {"final_selected_count": 8, "day_count": 1},
    ]
