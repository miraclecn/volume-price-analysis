from __future__ import annotations

import duckdb
import pandas as pd


def run_dimensions(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    columns = [
        "run_id",
        "experiment_name",
        "run_type",
        "status",
        "score_version",
        "created_at",
        "fold_count",
        "strategy_count",
        "score_count",
    ]
    if _table_exists(con, "ml_runs"):
        runs = _fetchdf(
            con,
            """
            select run_id, experiment_name, run_type, status, score_version, created_at
            from ml_runs
            """,
            columns[:6],
        )
    else:
        runs = pd.DataFrame(columns=columns[:6])

    dims = selection_options(con)
    if runs.empty and dims.empty:
        return pd.DataFrame(columns=columns)

    if runs.empty:
        runs = pd.DataFrame({"run_id": sorted(dims["run_id"].dropna().unique(), reverse=True)})
        runs["experiment_name"] = "legacy_backtest"
        runs["run_type"] = "backtest"
        runs["status"] = "legacy"
        runs["score_version"] = runs["run_id"].map(_first_score_by_run(dims))
        runs["created_at"] = ""
    else:
        missing = sorted(set(dims["run_id"].dropna()) - set(runs["run_id"].dropna()), reverse=True)
        if missing:
            legacy = pd.DataFrame({"run_id": missing})
            legacy["experiment_name"] = "legacy_backtest"
            legacy["run_type"] = "backtest"
            legacy["status"] = "legacy"
            legacy["score_version"] = legacy["run_id"].map(_first_score_by_run(dims))
            legacy["created_at"] = ""
            runs = pd.concat([runs, legacy], ignore_index=True)

    if dims.empty:
        runs["fold_count"] = 0
        runs["strategy_count"] = 0
        runs["score_count"] = 0
    else:
        counts = dims.groupby("run_id", dropna=True).agg(
            fold_count=("fold_id", lambda values: values.dropna().nunique()),
            strategy_count=("strategy_id", lambda values: values.dropna().nunique()),
            score_count=("score_version", lambda values: values.dropna().nunique()),
        )
        runs = runs.merge(counts, how="left", left_on="run_id", right_index=True)
        runs[["fold_count", "strategy_count", "score_count"]] = runs[["fold_count", "strategy_count", "score_count"]].fillna(0).astype(int)

    return runs[columns].sort_values(["created_at", "run_id"], ascending=[False, False], na_position="last").reset_index(drop=True)


def selection_options(con: duckdb.DuckDBPyConnection, run_id: str | None = None) -> pd.DataFrame:
    frames = [
        _dimension_options(con, "ml_backtest_nav"),
        _dimension_options(con, "ml_backtest_metrics"),
        _dimension_options(con, "ml_run_folds"),
    ]
    frames = [frame for frame in frames if not frame.empty]
    columns = ["run_id", "fold_id", "strategy_id", "score_version"]
    if not frames:
        return pd.DataFrame(columns=columns)
    result = pd.concat(frames, ignore_index=True).drop_duplicates()
    if run_id:
        result = result[result["run_id"] == run_id]
    return result.sort_values(["fold_id", "strategy_id", "score_version"], na_position="last").reset_index(drop=True)


def run_metadata(con: duckdb.DuckDBPyConnection, run_id: str) -> pd.DataFrame:
    return _fetchdf(
        con,
        "select * from ml_runs where run_id = ?",
        [
            "run_id",
            "run_type",
            "experiment_name",
            "config_path",
            "config_hash",
            "git_commit",
            "alpha_data_db",
            "alpha_data_latest_date",
            "vpa_db",
            "ml_db",
            "feature_set_id",
            "feature_store_version",
            "label_version",
            "score_version",
            "artifact_root",
            "created_at",
            "started_at",
            "finished_at",
            "status",
            "notes",
        ],
        [run_id],
    )


def run_folds(con: duckdb.DuckDBPyConnection, run_id: str) -> pd.DataFrame:
    return _fetchdf(
        con,
        """
        select *
        from ml_run_folds
        where run_id = ?
        order by test_start nulls last, fold_id
        """,
        [
            "run_id",
            "fold_id",
            "train_start",
            "train_end",
            "valid_start",
            "valid_end",
            "test_start",
            "test_end",
            "gap_type",
            "embargo_days",
            "status",
            "artifact_dir",
            "created_at",
        ],
        [run_id],
    )


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


def backtest_nav(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    fold_id: str | None = None,
    strategy_id: str | None = None,
    score_version: str | None = None,
) -> pd.DataFrame:
    columns = ["run_id", "fold_id", "strategy_id", "score_version", "sim_date", "nav", "cash", "gross_exposure", "turnover"]
    if not _table_exists(con, "ml_backtest_nav"):
        return pd.DataFrame(columns=columns)
    table_columns = _table_columns(con, "ml_backtest_nav")
    if (strategy_id and "strategy_id" not in table_columns) or (score_version and "score_version" not in table_columns):
        return pd.DataFrame(columns=columns)
    where, params = _dimension_filter(table_columns, run_id, fold_id, strategy_id, score_version)
    select_columns = [
        _column_or_null(table_columns, "run_id"),
        _column_or_null(table_columns, "fold_id"),
        _column_or_null(table_columns, "strategy_id"),
        _column_or_null(table_columns, "score_version"),
        _column_or_null(table_columns, "sim_date"),
        _column_or_null(table_columns, "nav"),
        _column_or_null(table_columns, "cash"),
        _column_or_null(table_columns, "gross_exposure"),
        _column_or_null(table_columns, "turnover"),
    ]
    return _fetchdf(
        con,
        f"""
        select {", ".join(select_columns)}
        from ml_backtest_nav
        {where}
        order by sim_date, fold_id
        """,
        columns,
        params,
    )


def fold_metric_matrix(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    fold_id: str | None = None,
    strategy_id: str | None = None,
    score_version: str | None = None,
) -> pd.DataFrame:
    columns = [
        "run_id",
        "fold_id",
        "strategy_id",
        "score_version",
        "annual_return",
        "max_drawdown",
        "calmar_like",
        "turnover",
        "empty_day_ratio",
    ]
    if not _table_exists(con, "ml_backtest_metrics"):
        return pd.DataFrame(columns=columns)
    table_columns = _table_columns(con, "ml_backtest_metrics")
    if (strategy_id and "strategy_id" not in table_columns) or (score_version and "score_version" not in table_columns):
        return pd.DataFrame(columns=columns)
    where, params = _dimension_filter(table_columns, run_id, fold_id, strategy_id, score_version)
    segment_clause = "and segment = 'fold'" if "segment" in table_columns else ""
    group_columns = [
        _column_or_null(table_columns, "run_id"),
        _column_or_null(table_columns, "fold_id"),
        _column_or_null(table_columns, "strategy_id"),
        _column_or_null(table_columns, "score_version"),
    ]
    return _fetchdf(
        con,
        f"""
        select
            {group_columns[0]},
            {group_columns[1]},
            {group_columns[2]},
            {group_columns[3]},
            max(case when metric_name in ('annual_return', 'annualized_return') then metric_value end) as annual_return,
            max(case when metric_name = 'max_drawdown' then metric_value end) as max_drawdown,
            max(case when metric_name in ('calmar', 'calmar_like') then metric_value end) as calmar_like,
            max(case when metric_name = 'turnover' then metric_value end) as turnover,
            max(case when metric_name = 'empty_day_ratio' then metric_value end) as empty_day_ratio
        from ml_backtest_metrics
        {where}
          {segment_clause}
        group by 1, 2, 3, 4
        order by fold_id, strategy_id, score_version
        """,
        columns,
        params,
    )


def continuous_nav(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    start_year: int = 2020,
    end_year: int = 2025,
    fold_suffix: str | None = None,
    strategy_id: str | None = None,
    score_version: str | None = None,
) -> pd.DataFrame:
    columns = [
        "run_id",
        "fold_id",
        "strategy_id",
        "score_version",
        "sim_date",
        "nav",
        "fold_return",
        "continuous_nav",
        "drawdown",
    ]
    if start_year > end_year:
        return pd.DataFrame(columns=columns)
    yearly_folds = [
        f"wf_{year}_{fold_suffix}" if fold_suffix else f"wf_{year}"
        for year in range(start_year, end_year + 1)
    ]
    nav = backtest_nav(
        con,
        run_id=run_id,
        strategy_id=None if fold_suffix else strategy_id,
        score_version=None if fold_suffix else score_version,
    )
    if nav.empty:
        return pd.DataFrame(columns=columns)
    nav = nav[nav["fold_id"].isin(yearly_folds)].copy()
    if nav.empty:
        return pd.DataFrame(columns=columns)
    nav["fold_order"] = nav["fold_id"].map({fold_id: idx for idx, fold_id in enumerate(yearly_folds)})
    nav["sim_date"] = nav["sim_date"].astype(str)
    nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
    nav = nav.dropna(subset=["nav"]).sort_values(["fold_order", "sim_date"]).reset_index(drop=True)
    if nav.empty:
        return pd.DataFrame(columns=columns)

    nav["fold_return"] = nav.groupby("fold_id", sort=False)["nav"].pct_change().fillna(0.0)
    initial_nav = float(nav["nav"].iloc[0])
    nav["continuous_nav"] = initial_nav * (1.0 + nav["fold_return"]).cumprod()
    nav["drawdown"] = nav["continuous_nav"].div(nav["continuous_nav"].cummax()).sub(1.0)
    return nav[columns]


def continuous_variant_options(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str | None = None,
    start_year: int = 2020,
    end_year: int = 2025,
) -> pd.DataFrame:
    columns = [
        "run_id",
        "fold_suffix",
        "score_version",
        "year_count",
        "min_annual_return",
        "mean_annual_return",
        "geometric_annual_return",
        "total_return",
    ]
    if start_year > end_year or not _table_exists(con, "ml_backtest_metrics"):
        return pd.DataFrame(columns=columns)
    table_columns = _table_columns(con, "ml_backtest_metrics")
    required = {"run_id", "fold_id", "metric_name", "metric_value"}
    if not required.issubset(table_columns):
        return pd.DataFrame(columns=columns)
    run_clause = "and run_id = ?" if run_id else ""
    params = [run_id] if run_id else []
    score_expr = _column_or_null(table_columns, "score_version")
    segment_clause = "and segment = 'fold'" if "segment" in table_columns else ""
    metrics = _fetchdf(
        con,
        f"""
        select
            run_id,
            fold_id,
            {score_expr},
            max(case when metric_name in ('annual_return', 'annualized_return') then metric_value end) as annual_return
        from ml_backtest_metrics
        where regexp_matches(fold_id, '^wf_[0-9]{{4}}_')
          {segment_clause}
          {run_clause}
        group by 1, 2, 3
        having annual_return is not null
        """,
        ["run_id", "fold_id", "score_version", "annual_return"],
        params,
    )
    if metrics.empty:
        return pd.DataFrame(columns=columns)

    parsed = metrics["fold_id"].astype(str).str.extract(r"^wf_(?P<year>\d{4})_(?P<fold_suffix>.+)$")
    metrics = pd.concat([metrics, parsed], axis=1).dropna(subset=["year", "fold_suffix"])
    metrics["year"] = metrics["year"].astype(int)
    metrics = metrics[(metrics["year"] >= int(start_year)) & (metrics["year"] <= int(end_year))]
    if metrics.empty:
        return pd.DataFrame(columns=columns)

    expected_years = set(range(int(start_year), int(end_year) + 1))
    rows: list[dict[str, object]] = []
    for (group_run_id, fold_suffix, score_version), group in metrics.groupby(["run_id", "fold_suffix", "score_version"], dropna=False):
        years = set(group["year"].dropna().astype(int))
        if not expected_years.issubset(years):
            continue
        selected = group[group["year"].isin(expected_years)].sort_values("year")
        annual_returns = pd.to_numeric(selected["annual_return"], errors="coerce").dropna()
        if annual_returns.empty:
            continue
        total_multiplier = float((1.0 + annual_returns).prod())
        rows.append(
            {
                "run_id": group_run_id,
                "fold_suffix": fold_suffix,
                "score_version": score_version,
                "year_count": len(expected_years),
                "min_annual_return": float(annual_returns.min()),
                "mean_annual_return": float(annual_returns.mean()),
                "geometric_annual_return": total_multiplier ** (1.0 / len(expected_years)) - 1.0,
                "total_return": total_multiplier - 1.0,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["min_annual_return", "geometric_annual_return", "total_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


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


def live_sim_accounts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return _fetchdf(
        con,
        """
        select account_id, initial_cash, created_at
        from live_sim_account
        order by account_id
        """,
        ["account_id", "initial_cash", "created_at"],
    )


def live_sim_nav(con: duckdb.DuckDBPyConnection, account_id: str | None = None) -> pd.DataFrame:
    where = "where account_id = ?" if account_id else ""
    params = [account_id] if account_id else []
    frame = _fetchdf(
        con,
        f"""
        select account_id, sim_date, nav, cash, holding_market_value, total_return, max_drawdown
        from live_sim_nav
        {where}
        order by case when sim_date = 'INITIAL' then 0 else 1 end, sim_date
        """,
        ["account_id", "sim_date", "nav", "cash", "holding_market_value", "total_return", "max_drawdown"],
        params,
    )
    if frame.empty:
        return frame.assign(daily_return=pd.Series(dtype=float), drawdown=pd.Series(dtype=float))
    out = frame.copy()
    out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
    out["daily_return"] = out.groupby("account_id", sort=False)["nav"].pct_change().fillna(0.0)
    out["drawdown"] = out.groupby("account_id", sort=False)["nav"].transform(lambda values: values / values.cummax() - 1.0)
    return out


def live_sim_order_summary(con: duckdb.DuckDBPyConnection, account_id: str | None = None) -> pd.DataFrame:
    account_filter = "where account_id = ?" if account_id else ""
    planned_params = [account_id] if account_id else []
    planned = _fetchdf(
        con,
        f"""
        select
            decision_date as sim_date,
            count(*) as planned_count,
            sum(case when side = 'buy' then 1 else 0 end) as planned_buy_count,
            sum(case when side = 'sell' then 1 else 0 end) as planned_sell_count
        from live_sim_planned_orders
        {account_filter}
        group by decision_date
        """,
        ["sim_date", "planned_count", "planned_buy_count", "planned_sell_count"],
        planned_params,
    )
    execution_filter = "where account_id = ?" if account_id else ""
    execution_params = [account_id] if account_id else []
    executions = _fetchdf(
        con,
        f"""
        select
            sim_date,
            count(*) as execution_count,
            sum(case when status = 'filled' then 1 else 0 end) as filled_count,
            sum(case when side = 'buy' and status = 'filled' then 1 else 0 end) as filled_buy_count,
            sum(case when side = 'sell' and status = 'filled' then 1 else 0 end) as filled_sell_count,
            sum(case when status = 'filled' then fees else 0 end) as fees
        from live_sim_executions
        {execution_filter}
        group by sim_date
        """,
        ["sim_date", "execution_count", "filled_count", "filled_buy_count", "filled_sell_count", "fees"],
        execution_params,
    )
    if planned.empty and executions.empty:
        return pd.DataFrame(
            columns=[
                "sim_date",
                "planned_count",
                "planned_buy_count",
                "planned_sell_count",
                "execution_count",
                "filled_count",
                "filled_buy_count",
                "filled_sell_count",
                "fees",
            ]
        )
    merged = planned.merge(executions, on="sim_date", how="outer").fillna(0)
    return merged.sort_values("sim_date").reset_index(drop=True)


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


def _dimension_filter(
    table_columns: set[str],
    run_id: str,
    fold_id: str | None = None,
    strategy_id: str | None = None,
    score_version: str | None = None,
) -> tuple[str, list[str]]:
    clauses = ["run_id = ?"]
    params = [run_id]
    if fold_id and "fold_id" in table_columns:
        clauses.append("fold_id = ?")
        params.append(fold_id)
    if strategy_id and "strategy_id" in table_columns:
        clauses.append("strategy_id = ?")
        params.append(strategy_id)
    if score_version and "score_version" in table_columns:
        clauses.append("score_version = ?")
        params.append(score_version)
    return "where " + " and ".join(clauses), params


def _dimension_options(con: duckdb.DuckDBPyConnection, table_name: str) -> pd.DataFrame:
    columns = ["run_id", "fold_id", "strategy_id", "score_version"]
    if not _table_exists(con, table_name):
        return pd.DataFrame(columns=columns)
    table_columns = _table_columns(con, table_name)
    if "run_id" not in table_columns:
        return pd.DataFrame(columns=columns)
    select_columns = [
        _column_or_null(table_columns, "run_id"),
        _column_or_null(table_columns, "fold_id"),
        _column_or_null(table_columns, "strategy_id"),
        _column_or_null(table_columns, "score_version"),
    ]
    return _fetchdf(
        con,
        f"""
        select distinct {", ".join(select_columns)}
        from {table_name}
        where run_id is not null
        """,
        columns,
    )


def _first_score_by_run(dimensions: pd.DataFrame) -> dict[str, str]:
    if dimensions.empty:
        return {}
    scores = dimensions.dropna(subset=["score_version"])
    if scores.empty:
        return {}
    return scores.groupby("run_id")["score_version"].first().to_dict()


def _column_or_null(table_columns: set[str], column: str) -> str:
    if column in table_columns:
        return column
    return f"null::varchar as {column}"


def _table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = con.execute(
        "select count(*) from information_schema.tables where table_schema = 'main' and table_name = ?",
        [table_name],
    ).fetchone()
    return bool(row and row[0])


def _table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    return {
        row[0]
        for row in con.execute(
            """
            select column_name
            from information_schema.columns
            where table_schema = 'main' and table_name = ?
            """,
            [table_name],
        ).fetchall()
    }


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
