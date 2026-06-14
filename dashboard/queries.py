from __future__ import annotations

import duckdb
import pandas as pd


def run_registry(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return _fetchdf(
        con,
        """
        with metric_pivot as (
            select
                run_id,
                avg(case when metric_name = 'annual_return' and segment = 'fold' then metric_value end) as annual_return_mean,
                min(case when metric_name = 'max_drawdown' and segment = 'fold' then metric_value end) as max_drawdown_worst,
                avg(case when metric_name = 'annual_return' and segment = 'fold' and metric_value > 0 then 1.0
                         when metric_name = 'annual_return' and segment = 'fold' then 0.0 end) as positive_year_ratio
            from ml_backtest_metrics
            group by run_id
        )
        select
            r.run_id,
            r.experiment_name,
            r.run_type,
            r.status,
            r.feature_set_id,
            r.label_version,
            r.score_version,
            r.config_hash,
            r.git_commit,
            r.created_at,
            coalesce(m.annual_return_mean, 0.0) as annual_return_mean,
            coalesce(m.max_drawdown_worst, 0.0) as max_drawdown_worst,
            coalesce(m.positive_year_ratio, 0.0) as positive_year_ratio
        from ml_runs r
        left join metric_pivot m using (run_id)
        order by r.created_at desc nulls last, r.run_id
        """,
        [
            "run_id",
            "experiment_name",
            "run_type",
            "status",
            "feature_set_id",
            "label_version",
            "score_version",
            "config_hash",
            "git_commit",
            "created_at",
            "annual_return_mean",
            "max_drawdown_worst",
            "positive_year_ratio",
        ],
    )


def walkforward_compare(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return _fetchdf(
        con,
        """
        select run_id, strategy_id, score_version, metric_name, metric_value
        from ml_backtest_metrics
        where segment = 'walkforward'
          and metric_name in (
              'mean_annual_return', 'median_annual_return', 'worst_year_return',
              'max_of_max_drawdown', 'mean_calmar', 'high_return_capture_ratio'
          )
        order by run_id, strategy_id, score_version, metric_name
        """,
        ["run_id", "strategy_id", "score_version", "metric_name", "metric_value"],
    )


def score_mode_compare(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return _fetchdf(
        con,
        """
        select run_id, fold_id, strategy_id, score_version, metric_name, metric_value
        from ml_backtest_metrics
        where segment = 'fold'
          and score_version in ('v2_three_model', 'v2_absolute_only', 'v2_absolute_risk_filter', 'v2_absolute_risk_sort')
          and metric_name in ('annual_return', 'max_drawdown', 'calmar', 'calmar_like')
        order by run_id, fold_id, strategy_id, score_version, metric_name
        """,
        ["run_id", "fold_id", "strategy_id", "score_version", "metric_name", "metric_value"],
    )


def fixed_horizon_compare(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return _fetchdf(
        con,
        """
        select run_id, fold_id, strategy_id, score_version, metric_name, metric_value, segment
        from ml_backtest_metrics
        where strategy_id in ('holding_aware_v2', 'abs_ranker_fixed_5d_risk_filter_v1', 'abs_ranker_fixed_5d_no_risk_exit_v1')
           or metric_name in ('risk_exit_benefit', 'fixed_horizon_vs_holding_aware_delta', 'fixed_horizon_vs_holding_aware_drawdown_delta')
        order by run_id, fold_id, segment, strategy_id, metric_name
        """,
        ["run_id", "fold_id", "strategy_id", "score_version", "metric_name", "metric_value", "segment"],
    )


def fold_detail(con: duckdb.DuckDBPyConnection, run_id: str | None = None, fold_id: str | None = None) -> dict[str, pd.DataFrame]:
    where, params = _run_fold_filter(run_id, fold_id)
    return {
        "nav": _fetchdf(con, f"select * from ml_backtest_nav {where} order by sim_date", ["sim_date", "nav"], params),
        "orders": _fetchdf(con, f"select * from ml_backtest_orders {where} order by sim_date, code", ["sim_date", "code", "side", "status"], params),
        "positions": _fetchdf(con, f"select * from ml_backtest_positions {where} order by sim_date, code", ["sim_date", "code", "weight"], params),
        "diagnostics": _fetchdf(con, _diagnostics_query(run_id, fold_id), ["trade_date", "run_id", "fold_id", "portfolio_id"], params),
    }


def model_bundle_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return _fetchdf(
        con,
        """
        select
            bundle_id,
            run_id,
            fold_id,
            bundle_role,
            absolute_model_id,
            active_model_id,
            risk_model_id,
            feature_set_id,
            label_base,
            horizon_d,
            score_version,
            artifact_dir,
            status,
            created_at
        from ml_model_bundles
        order by status desc nulls last, created_at desc nulls last, bundle_id
        """,
        [
            "bundle_id",
            "run_id",
            "fold_id",
            "bundle_role",
            "absolute_model_id",
            "active_model_id",
            "risk_model_id",
            "feature_set_id",
            "label_base",
            "horizon_d",
            "score_version",
            "artifact_dir",
            "status",
            "created_at",
        ],
    )


def portfolio_diagnostics(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return _fetchdf(
        con,
        """
        select *
        from ml_portfolio_construction_diagnostics
        order by trade_date desc, run_id, fold_id, portfolio_id
        limit 5000
        """,
        ["trade_date", "run_id", "fold_id", "portfolio_id", "score_version"],
    )


def signal_preview(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return _fetchdf(
        con,
        """
        select
            p.trade_date,
            p.code,
            pred.absolute_score,
            pred.active_score,
            pred.risk_prob,
            pred.trade_score_v2,
            p.target_weight,
            p.reason,
            p.source_sleeve,
            p.source_bundle_id,
            p.score_version
        from live_target_positions p
        left join ml_predictions_daily pred
          on pred.trade_date = p.trade_date
         and pred.code = p.code
         and pred.score_version = p.score_version
        order by p.trade_date desc, p.source_sleeve, p.target_weight desc
        limit 5000
        """,
        [
            "trade_date",
            "code",
            "absolute_score",
            "active_score",
            "risk_prob",
            "trade_score_v2",
            "target_weight",
            "reason",
            "source_sleeve",
            "source_bundle_id",
            "score_version",
        ],
    )


def live_monitor(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return _fetchdf(
        con,
        """
        select
            coalesce(t.trade_date, o.trade_date) as trade_date,
            coalesce(t.account_id, o.account_id) as account_id,
            coalesce(t.strategy_id, o.strategy_id) as strategy_id,
            count(distinct t.code) as target_count,
            count(distinct o.order_id) as order_count,
            count(distinct f.fill_id) as fill_count,
            avg(f.slippage_bps) as avg_slippage_bps
        from live_target_positions t
        full outer join live_orders o
          on o.account_id = t.account_id
         and o.strategy_id = t.strategy_id
         and o.code = t.code
        left join live_fills f using (order_id)
        group by 1, 2, 3
        order by trade_date desc, account_id, strategy_id
        """,
        ["trade_date", "account_id", "strategy_id", "target_count", "order_count", "fill_count", "avg_slippage_bps"],
    )


def data_health_summary(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    table_count = con.execute(
        "select count(*) from information_schema.tables where table_schema = 'main'"
    ).fetchone()[0]
    latest_trade_date = _safe_scalar(con, "select max(trade_date) from ml_tradeability_daily")
    prediction_rows = _safe_scalar(con, "select count(*) from ml_predictions_daily") or 0
    unknown_ratio = _safe_scalar(
        con,
        """
        select avg(case when industry_code is null or industry_code in ('UNKNOWN', '') then 1.0 else 0.0 end)
        from ml_tradeability_daily
        """,
    )
    return {
        "table_count": int(table_count),
        "latest_trade_date": str(latest_trade_date) if latest_trade_date is not None else "",
        "prediction_rows": int(prediction_rows),
        "unknown_industry_ratio": float(unknown_ratio or 0.0),
    }


def _run_fold_filter(run_id: str | None, fold_id: str | None) -> tuple[str, list[str]]:
    clauses: list[str] = []
    params: list[str] = []
    if run_id is not None:
        clauses.append("run_id = ?")
        params.append(run_id)
    if fold_id is not None:
        clauses.append("fold_id = ?")
        params.append(fold_id)
    return ("where " + " and ".join(clauses), params) if clauses else ("", params)


def _diagnostics_query(run_id: str | None, fold_id: str | None) -> str:
    where, _ = _run_fold_filter(run_id, fold_id)
    return f"select * from ml_portfolio_construction_diagnostics {where} order by trade_date"


def _safe_scalar(con: duckdb.DuckDBPyConnection, sql: str) -> object | None:
    try:
        row = con.execute(sql).fetchone()
    except Exception:
        return None
    return row[0] if row else None


def _fetchdf(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    columns: list[str],
    params: list[str] | None = None,
) -> pd.DataFrame:
    try:
        return con.execute(sql, params or []).fetchdf()
    except (duckdb.CatalogException, duckdb.BinderException):
        return pd.DataFrame(columns=columns)
