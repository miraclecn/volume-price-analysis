from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd


def init_ml_db(path: Path | str) -> duckdb.DuckDBPyConnection:
    db_path = Path(path)
    if str(path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    schema_path = Path(__file__).resolve().parents[1] / "sql" / "create_ml_tables.sql"
    con.execute(schema_path.read_text(encoding="utf-8"))
    _apply_migrations(con)
    return con


def _apply_migrations(con: duckdb.DuckDBPyConnection) -> None:
    alters = [
        "alter table ml_tradeability_daily add column if not exists is_bse boolean",
        "alter table ml_feature_mart_daily add column if not exists is_bse boolean",
        "alter table ml_predictions_daily add column if not exists run_id varchar",
        "alter table ml_predictions_daily add column if not exists fold_id varchar",
        "alter table ml_predictions_daily add column if not exists absolute_model_id varchar",
        "alter table ml_predictions_daily add column if not exists active_model_id varchar",
        "alter table ml_predictions_daily add column if not exists risk_model_id varchar",
        "alter table ml_model_registry add column if not exists train_start varchar",
        "alter table ml_model_registry add column if not exists run_id varchar",
        "alter table ml_model_registry add column if not exists fold_id varchar",
        "alter table ml_model_registry add column if not exists feature_store_version varchar",
        "alter table ml_model_registry add column if not exists feature_schema_hash varchar",
        "alter table ml_model_registry add column if not exists train_end varchar",
        "alter table ml_model_registry add column if not exists valid_start varchar",
        "alter table ml_model_registry add column if not exists valid_end varchar",
        "alter table ml_model_registry add column if not exists test_start varchar",
        "alter table ml_model_registry add column if not exists test_end varchar",
        "alter table ml_backtest_metrics add column if not exists fold_id varchar",
        "alter table ml_backtest_metrics add column if not exists strategy_id varchar",
        "alter table ml_backtest_metrics add column if not exists score_version varchar",
        "alter table ml_backtest_metrics add column if not exists start_date varchar",
        "alter table ml_backtest_metrics add column if not exists end_date varchar",
        "alter table ml_backtest_metrics add column if not exists annual_return double",
        "alter table ml_backtest_metrics add column if not exists total_return double",
        "alter table ml_backtest_metrics add column if not exists max_drawdown double",
        "alter table ml_backtest_metrics add column if not exists calmar_like double",
        "alter table ml_backtest_metrics add column if not exists turnover double",
        "alter table ml_backtest_metrics add column if not exists win_rate double",
        "alter table ml_backtest_metrics add column if not exists empty_day_ratio double",
        "alter table ml_backtest_metrics add column if not exists cash_ratio_avg double",
        "alter table ml_backtest_metrics add column if not exists bse_excluded_count double",
        "alter table ml_backtest_metrics add column if not exists unknown_industry_weight_avg double",
        "alter table ml_backtest_metrics add column if not exists core_pool_size_avg double",
        "alter table ml_backtest_metrics add column if not exists candidate_pool_size_avg double",
        "alter table ml_backtest_metrics add column if not exists avg_holding_days double",
        "alter table ml_backtest_metrics add column if not exists median_holding_days double",
        "alter table ml_backtest_metrics add column if not exists max_holding_days double",
        "alter table ml_backtest_metrics add column if not exists holding_segment_count double",
        "alter table ml_backtest_metrics add column if not exists turnover_daily_avg double",
        "alter table ml_backtest_metrics add column if not exists sell_blocked_count double",
        "alter table ml_backtest_orders add column if not exists entry_date varchar",
        "alter table ml_backtest_orders add column if not exists exit_date varchar",
        "alter table ml_backtest_orders add column if not exists holding_days integer",
        "alter table ml_backtest_orders add column if not exists entry_trade_score double",
        "alter table ml_backtest_orders add column if not exists exit_trade_score double",
        "alter table ml_backtest_orders add column if not exists entry_reason varchar",
        "alter table ml_backtest_orders add column if not exists exit_reason varchar",
        "alter table ml_backtest_orders add column if not exists sell_blocked_reason varchar",
        "alter table ml_backtest_positions add column if not exists entry_date varchar",
        "alter table ml_backtest_positions add column if not exists entry_price double",
        "alter table ml_backtest_positions add column if not exists holding_days integer",
        "alter table ml_backtest_positions add column if not exists entry_trade_score double",
        "alter table ml_backtest_positions add column if not exists entry_reason varchar",
        "alter table ml_portfolio_targets_daily add column if not exists hold_reason varchar",
        "alter table ml_portfolio_targets_daily add column if not exists signal_action varchar",
        "alter table ml_portfolio_targets_daily add column if not exists exit_reason varchar",
        "alter table ml_portfolio_targets_daily add column if not exists sell_blocked_reason varchar",
        "alter table ml_portfolio_targets_daily add column if not exists entry_date varchar",
        "alter table ml_portfolio_targets_daily add column if not exists entry_price double",
        "alter table ml_portfolio_targets_daily add column if not exists shares double",
        "alter table ml_portfolio_targets_daily add column if not exists holding_days integer",
        "alter table ml_portfolio_targets_daily add column if not exists entry_trade_score double",
        "alter table ml_portfolio_targets_daily add column if not exists latest_trade_score double",
        "alter table ml_portfolio_construction_diagnostics add column if not exists retained_holdings_count integer",
        "alter table ml_portfolio_construction_diagnostics add column if not exists sell_signal_count integer",
        "alter table ml_portfolio_construction_diagnostics add column if not exists sell_executed_count integer",
        "alter table ml_portfolio_construction_diagnostics add column if not exists sell_blocked_count integer",
        "alter table ml_portfolio_construction_diagnostics add column if not exists hold_due_to_min_days_count integer",
        "alter table ml_portfolio_construction_diagnostics add column if not exists hold_due_to_score_ok_count integer",
        "alter table ml_portfolio_construction_diagnostics add column if not exists exit_due_to_score_count integer",
        "alter table ml_portfolio_construction_diagnostics add column if not exists exit_due_to_risk_count integer",
        "alter table ml_portfolio_construction_diagnostics add column if not exists exit_due_to_time_count integer",
        "alter table ml_portfolio_construction_diagnostics add column if not exists exit_due_to_not_candidate_count integer",
        "alter table ml_portfolio_construction_diagnostics add column if not exists avg_holding_days_current double",
        "alter table ml_portfolio_construction_diagnostics add column if not exists median_holding_days_current double",
        """
        create table if not exists ml_prediction_raw_daily (
            trade_date varchar not null,
            code varchar not null,
            run_id varchar not null,
            fold_id varchar not null,
            score_version varchar not null,
            feature_set_id varchar not null,
            horizon_d integer not null,
            absolute_model_id varchar,
            active_model_id varchar,
            risk_model_id varchar,
            absolute_score double,
            active_score double,
            risk_prob double,
            generated_at varchar not null,
            primary key (trade_date, code, run_id, fold_id, horizon_d)
        )
        """,
        """
        create table if not exists ml_portfolio_construction_diagnostics (
            trade_date varchar not null,
            run_id varchar not null,
            fold_id varchar not null,
            portfolio_id varchar not null,
            score_version varchar not null,
            raw_candidate_count integer,
            hard_filter_pass_count integer,
            core_pool_size integer,
            candidate_pool_size integer,
            selected_from_core integer,
            selected_from_candidate integer,
            final_selected_count integer,
            low_adv_rejected_count integer,
            cannot_buy_rejected_count integer,
            st_rejected_count integer,
            paused_rejected_count integer,
            bse_rejected_count integer,
            low_trade_score_rejected_count integer,
            high_risk_rejected_count integer,
            industry_limit_blocked_count integer,
            unknown_industry_limit_blocked_count integer,
            max_new_entries_blocked_count integer,
            cash_weight double,
            created_at varchar,
            primary key (trade_date, run_id, fold_id, portfolio_id, score_version)
        )
        """,
    ]
    for sql in alters:
        try:
            con.execute(sql)
        except Exception:
            # Keep startup resilient for historical table variants.
            continue
    _migrate_backtest_metrics_primary_key(con)


def _migrate_backtest_metrics_primary_key(con: duckdb.DuckDBPyConnection) -> None:
    row = con.execute(
        "select sql from duckdb_tables() where table_name = 'ml_backtest_metrics'"
    ).fetchone()
    if row is None:
        return
    table_sql = str(row[0]).lower().replace(" ", "")
    if "primarykey(run_id,fold_id,score_version,metric_name,segment)" in table_sql:
        return
    temp_name = f"_ml_backtest_metrics_migrate_{uuid4().hex}"
    con.execute(
        f"""
        create table {temp_name} (
            run_id varchar not null,
            fold_id varchar,
            strategy_id varchar,
            score_version varchar,
            start_date varchar,
            end_date varchar,
            metric_name varchar not null,
            metric_value double,
            annual_return double,
            total_return double,
            max_drawdown double,
            calmar_like double,
            turnover double,
            win_rate double,
            empty_day_ratio double,
            cash_ratio_avg double,
            bse_excluded_count double,
            unknown_industry_weight_avg double,
            core_pool_size_avg double,
            candidate_pool_size_avg double,
            avg_holding_days double,
            median_holding_days double,
            max_holding_days double,
            holding_segment_count double,
            turnover_daily_avg double,
            sell_blocked_count double,
            segment varchar not null,
            primary key (run_id, fold_id, score_version, metric_name, segment)
        )
        """
    )
    con.execute(
        f"""
        insert into {temp_name} by name
        select *
        from ml_backtest_metrics
        """
    )
    con.execute("drop table ml_backtest_metrics")
    con.execute(f"alter table {temp_name} rename to ml_backtest_metrics")


def upsert_dataframe(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
    key_columns: list[str],
) -> None:
    if frame.empty:
        return
    table_columns = [
        row[0]
        for row in con.execute(
            """
            select column_name
            from information_schema.columns
            where table_schema = 'main' and table_name = ?
            order by ordinal_position
            """,
            [table_name],
        ).fetchall()
    ]
    frame = frame[[column for column in table_columns if column in frame.columns]].copy()
    temp_name = f"_ml_upsert_{uuid4().hex}"
    con.register(temp_name, frame)
    condition = " and ".join(
        f"{table_name}.{column} = {temp_name}.{column}" for column in key_columns
    )
    try:
        con.execute(f"delete from {table_name} using {temp_name} where {condition}")
        con.execute(f"insert into {table_name} by name select * from {temp_name}")
    finally:
        con.unregister(temp_name)
