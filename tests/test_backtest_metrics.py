from __future__ import annotations

import pandas as pd

from ml_stock_selector.backtest.metrics import (
    annualized_return,
    cash_days_ratio,
    compare_backtest_metric_rows,
    max_drawdown,
    ndcg_at_k,
    pool_size_metrics,
    holding_period_metrics,
    rank_ic,
    summarize_fold_metric_rows,
    summarize_walkforward_metric_rows,
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


def test_summarize_fold_metric_rows_include_phase8_standard_risk_metrics():
    result = BacktestResult(
        orders=pd.DataFrame({"status": ["filled"], "side": ["buy"]}),
        positions=pd.DataFrame(
            {
                "sim_date": ["2024-01-02", "2024-01-02", "2024-02-02"],
                "code": ["a", "b", "a"],
                "industry_code": ["I1", "I2", "I1"],
                "weight": [0.4, 0.3, 0.5],
            }
        ),
        nav=pd.DataFrame(
            {
                "sim_date": ["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"],
                "nav": [100.0, 110.0, 99.0, 120.0],
                "turnover": [0.0, 0.1, 0.0, 0.2],
                "gross_exposure": [0.4, 0.7, 0.0, 0.5],
            }
        ),
    )

    rows = summarize_fold_metric_rows(
        result,
        run_id="run",
        fold_id="wf_2024",
        score_version="v2_three_model",
        strategy_id="holding_aware_v2",
        start_date="2024-01-02",
        end_date="2024-02-02",
        candidate_pool_size=10,
        core_pool_size=5,
    )
    values = dict(zip(rows["metric_name"], rows["metric_value"], strict=False))

    assert {
        "sharpe",
        "sortino",
        "volatility",
        "win_rate_daily",
        "win_rate_monthly",
        "position_count_avg",
        "best_month",
        "worst_month",
        "max_consecutive_loss_days",
        "max_consecutive_loss_months",
    }.issubset(values)
    assert values["position_count_avg"] == 1.5
    assert values["best_month"] > 0
    assert values["worst_month"] < 0
    assert values["max_consecutive_loss_days"] == 1.0


def test_summarize_walkforward_metric_rows_captures_worst_year_and_high_return():
    metrics = pd.DataFrame(
        [
            {"run_id": "run", "fold_id": "wf_2020", "score_version": "v2_three_model", "metric_name": "annual_return", "metric_value": 1.20, "segment": "fold"},
            {"run_id": "run", "fold_id": "wf_2020", "score_version": "v2_three_model", "metric_name": "max_drawdown", "metric_value": -0.18, "segment": "fold"},
            {"run_id": "run", "fold_id": "wf_2020", "score_version": "v2_three_model", "metric_name": "calmar_like", "metric_value": 6.6, "segment": "fold"},
            {"run_id": "run", "fold_id": "wf_2021", "score_version": "v2_three_model", "metric_name": "annual_return", "metric_value": -0.10, "segment": "fold"},
            {"run_id": "run", "fold_id": "wf_2021", "score_version": "v2_three_model", "metric_name": "max_drawdown", "metric_value": -0.35, "segment": "fold"},
            {"run_id": "run", "fold_id": "wf_2021", "score_version": "v2_three_model", "metric_name": "calmar_like", "metric_value": -0.3, "segment": "fold"},
        ]
    )

    rows = summarize_walkforward_metric_rows(metrics, run_id="run", score_version="v2_three_model")
    values = dict(zip(rows["metric_name"], rows["metric_value"], strict=False))

    assert values["mean_annual_return"] == 0.55
    assert values["negative_year_count"] == 1.0
    assert values["drawdown_over_30_count"] == 1.0
    assert values["high_return_capture_ratio"] == 1.0
    assert values["worst_year_return"] == -0.10
    assert values["best_year_return"] == 1.20
    assert rows["segment"].eq("walkforward").all()


def test_compare_backtest_metric_rows_reports_score_and_fixed_horizon_deltas():
    metrics = pd.DataFrame(
        [
            {"run_id": "run", "fold_id": "wf_2024", "strategy_id": "holding_aware_v2", "score_version": "v2_three_model", "metric_name": "annual_return", "metric_value": 0.40, "segment": "fold"},
            {"run_id": "run", "fold_id": "wf_2024", "strategy_id": "holding_aware_v2", "score_version": "v2_three_model", "metric_name": "max_drawdown", "metric_value": -0.20, "segment": "fold"},
            {"run_id": "run", "fold_id": "wf_2024", "strategy_id": "holding_aware_v2", "score_version": "v2_absolute_only", "metric_name": "annual_return", "metric_value": 0.30, "segment": "fold"},
            {"run_id": "run", "fold_id": "wf_2024", "strategy_id": "holding_aware_v2", "score_version": "v2_absolute_only", "metric_name": "max_drawdown", "metric_value": -0.25, "segment": "fold"},
            {"run_id": "run", "fold_id": "wf_2024", "strategy_id": "holding_aware_v2", "score_version": "v2_absolute_risk_filter", "metric_name": "annual_return", "metric_value": 0.35, "segment": "fold"},
            {"run_id": "run", "fold_id": "wf_2024", "strategy_id": "holding_aware_v2", "score_version": "v2_absolute_risk_filter", "metric_name": "max_drawdown", "metric_value": -0.18, "segment": "fold"},
            {"run_id": "run", "fold_id": "wf_2024", "strategy_id": "holding_aware_v2", "score_version": "v2_absolute_risk_sort", "metric_name": "annual_return", "metric_value": 0.37, "segment": "fold"},
            {"run_id": "run", "fold_id": "wf_2024", "strategy_id": "holding_aware_v2", "score_version": "v2_absolute_risk_sort", "metric_name": "max_drawdown", "metric_value": -0.17, "segment": "fold"},
            {"run_id": "run", "fold_id": "wf_2024", "strategy_id": "abs_ranker_fixed_5d_risk_filter_v1", "score_version": "abs_ranker_fixed_5d_risk_filter_v1", "metric_name": "annual_return", "metric_value": 0.28, "segment": "fold"},
            {"run_id": "run", "fold_id": "wf_2024", "strategy_id": "abs_ranker_fixed_5d_no_risk_exit_v1", "score_version": "abs_ranker_fixed_5d_no_risk_exit_v1", "metric_name": "annual_return", "metric_value": 0.32, "segment": "fold"},
        ]
    )

    rows = compare_backtest_metric_rows(metrics, run_id="run")
    values = dict(zip(rows["metric_name"], rows["metric_value"], strict=False))

    assert values["absolute_only_vs_three_model_delta"] == -0.10
    assert values["risk_filter_return_delta"] == 0.05
    assert values["absolute_risk_sort_vs_risk_filter_delta"] == 0.02
    assert values["risk_exit_benefit"] == -0.04
    assert values["fixed_horizon_vs_holding_aware_delta"] == -0.12
