from __future__ import annotations

import pandas as pd

from ml_stock_selector.backtest.metrics import (
    annualized_return,
    cash_days_ratio,
    max_drawdown,
    ndcg_at_k,
    pool_size_metrics,
    holding_period_metrics,
    rank_ic,
    summarize_fold_metric_rows,
    unknown_industry_daily_exposure,
)
from ml_stock_selector.backtest.engine import BacktestResult
from ml_stock_selector.backtest.reports import prediction_report_metrics


def test_backtest_metrics_match_small_fixture():
    nav = pd.DataFrame({"nav": [1.0, 1.1, 1.05]})
    preds = pd.DataFrame({"trade_date": ["d"] * 3, "alpha_score": [0.9, 0.5, 0.1], "future_score": [3.0, 2.0, 1.0]})

    assert max_drawdown(nav) < 0
    assert annualized_return(nav) != 0
    assert rank_ic(preds, "alpha_score", "future_score") > 0
    assert ndcg_at_k(preds, "alpha_score", "future_score", 2) > 0
    assert cash_days_ratio(pd.DataFrame({"gross_exposure": [0.0, 0.5, 0.0]})) == 2 / 3


def test_backtest_metrics_sort_nav_by_date_before_drawdown():
    nav = pd.DataFrame(
        {
            "sim_date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "nav": [90.0, 100.0, 120.0],
        }
    )

    assert max_drawdown(nav) == -0.25


def test_annualized_return_uses_observed_date_span_not_row_frequency():
    nav = pd.DataFrame(
        {
            "sim_date": ["2024-01-01", "2024-07-01", "2024-12-31"],
            "nav": [100.0, 101.0, 102.0],
        }
    )

    assert 0.019 <= annualized_return(nav) <= 0.021


def test_unknown_industry_daily_exposure_counts_positions_and_weight():
    positions = pd.DataFrame(
        {
            "sim_date": ["2024-01-03", "2024-01-03"],
            "code": ["u1", "i1"],
            "industry_code": ["UNKNOWN", "I1"],
            "weight": [0.2, 0.3],
        }
    )

    exposure = unknown_industry_daily_exposure(positions)

    assert exposure["unknown_industry_position_count"].tolist() == [1]
    assert exposure["unknown_industry_weight"].tolist() == [0.2]


def test_prediction_report_metrics_summarize_v2_scores():
    predictions = pd.DataFrame(
        {
            "model_id": ["m1", "m1", "m2"],
            "score_version": ["v2_three_model", "v2_three_model", "v1_legacy"],
            "active_rank_pct": [0.8, 0.6, None],
            "risk_prob": [0.1, 0.3, None],
        }
    )

    metrics = prediction_report_metrics(predictions)

    assert metrics["prediction_model_count"] == 2.0
    assert metrics["prediction_v2_three_model_rows"] == 2.0
    assert metrics["prediction_active_rank_pct_mean"] == 0.7
    assert metrics["prediction_risk_prob_mean"] == 0.2


def test_pool_size_metrics_counts_candidate_and_core_rows():
    metrics = pool_size_metrics(
        pd.DataFrame({"code": ["a", "b", "c"]}),
        pd.DataFrame({"code": ["a"]}),
    )

    assert metrics["candidate_pool_size"] == 3.0
    assert metrics["core_pool_size"] == 1.0


def test_holding_period_metrics_summarize_filled_sells_and_blocked_sells():
    metrics = holding_period_metrics(
        pd.DataFrame(
            {
                "side": ["sell", "sell", "buy"],
                "status": ["filled", "rejected", "filled"],
                "holding_days": [5, None, None],
            }
        ),
        pd.DataFrame({"turnover": [0.1, 0.3]}),
    )

    assert metrics["avg_holding_days"] == 5.0
    assert metrics["median_holding_days"] == 5.0
    assert metrics["max_holding_days"] == 5.0
    assert metrics["holding_segment_count"] == 1.0
    assert metrics["turnover_daily_avg"] == 0.2
    assert metrics["sell_blocked_count"] == 1.0


def test_summarize_fold_metric_rows_include_p1_audit_metrics():
    result = BacktestResult(
        orders=pd.DataFrame({"status": ["filled", "rejected"], "side": ["buy", "buy"]}),
        positions=pd.DataFrame(
            {
                "sim_date": ["2024-01-03", "2024-01-04"],
                "code": ["a", "b"],
                "industry_code": ["UNKNOWN", "I1"],
                "weight": [0.2, 0.3],
            }
        ),
        nav=pd.DataFrame(
            {
                "sim_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "nav": [100.0, 105.0, 103.0],
                "turnover": [0.0, 0.1, 0.2],
                "gross_exposure": [0.0, 0.5, 0.7],
            }
        ),
    )

    rows = summarize_fold_metric_rows(
        result,
        run_id="run",
        fold_id="wf_2020",
        score_version="v2_three_model",
        strategy_id="p_v2",
        start_date="2024-01-02",
        end_date="2024-01-04",
        candidate_pool_size=10,
        core_pool_size=3,
        bse_excluded_count=2,
    )

    metric_names = set(rows["metric_name"])
    assert {
        "annual_return",
        "total_return",
        "max_drawdown",
        "calmar_like",
        "turnover",
        "win_rate",
        "empty_day_ratio",
        "cash_ratio_avg",
        "bse_excluded_count",
        "unknown_industry_weight_avg",
        "core_pool_size_avg",
        "candidate_pool_size_avg",
        "avg_holding_days",
        "median_holding_days",
        "max_holding_days",
        "holding_segment_count",
        "turnover_daily_avg",
        "sell_blocked_count",
    }.issubset(metric_names)
    assert rows["strategy_id"].eq("p_v2").all()
    assert rows["start_date"].eq("2024-01-02").all()
