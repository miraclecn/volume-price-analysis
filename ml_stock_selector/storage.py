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
        "alter table ml_backtest_orders add column if not exists entry_price double",
        "alter table ml_backtest_orders add column if not exists fold_id varchar",
        "alter table ml_backtest_orders add column if not exists exit_date varchar",
        "alter table ml_backtest_orders add column if not exists holding_days integer",
        "alter table ml_backtest_orders add column if not exists entry_trade_score double",
        "alter table ml_backtest_orders add column if not exists exit_trade_score double",
        "alter table ml_backtest_orders add column if not exists entry_reason varchar",
        "alter table ml_backtest_orders add column if not exists exit_reason varchar",
        "alter table ml_backtest_orders add column if not exists sell_blocked_reason varchar",
        "alter table ml_backtest_orders add column if not exists entry_abs_rank_pct double",
        "alter table ml_backtest_orders add column if not exists entry_risk_rank_pct double",
        "alter table ml_backtest_orders add column if not exists strategy_id varchar",
        "alter table ml_backtest_orders add column if not exists score_version varchar",
        "alter table ml_backtest_orders add column if not exists order_seq integer",
        "alter table ml_backtest_orders add column if not exists realized_pnl double",
        "alter table ml_backtest_positions add column if not exists fold_id varchar",
        "alter table ml_backtest_positions add column if not exists strategy_id varchar",
        "alter table ml_backtest_positions add column if not exists score_version varchar",
        "alter table ml_backtest_positions add column if not exists entry_date varchar",
        "alter table ml_backtest_positions add column if not exists entry_price double",
        "alter table ml_backtest_positions add column if not exists holding_days integer",
        "alter table ml_backtest_positions add column if not exists entry_trade_score double",
        "alter table ml_backtest_positions add column if not exists entry_reason varchar",
        "alter table ml_backtest_nav add column if not exists fold_id varchar",
        "alter table ml_backtest_nav add column if not exists strategy_id varchar",
        "alter table ml_backtest_nav add column if not exists score_version varchar",
        "alter table ml_portfolio_targets_daily add column if not exists run_id varchar",
        "alter table ml_portfolio_targets_daily add column if not exists fold_id varchar",
        "alter table ml_portfolio_targets_daily add column if not exists score_version varchar",
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
        create table if not exists ml_market_regime_daily (
            trade_date varchar primary key,
            trend_score double,
            breadth_score double,
            sentiment_score double,
            liquidity_score double,
            volatility_score double,
            final_regime varchar,
            generated_at varchar
        )
        """,
        """
        create table if not exists ml_model_health_daily (
            trade_date varchar not null,
            model_or_bundle_id varchar not null,
            strategy_id varchar,
            score_version varchar,
            rolling_20d_return double,
            rolling_60d_return double,
            rolling_20d_drawdown double,
            rolling_60d_drawdown double,
            equity_above_ma60 boolean,
            enabled_by_health boolean,
            reason varchar,
            primary key (trade_date, model_or_bundle_id, strategy_id, score_version)
        )
        """,
        """
        create table if not exists ml_strategy_allocation_daily (
            trade_date varchar not null,
            strategy_id varchar not null,
            sleeve varchar not null,
            bundle_id varchar,
            score_version varchar,
            raw_weight double,
            regime_multiplier double,
            health_multiplier double,
            drawdown_multiplier double,
            final_weight double,
            reason varchar,
            generated_at varchar,
            primary key (trade_date, strategy_id, sleeve, bundle_id, score_version)
        )
        """,
        """
        create table if not exists live_target_positions (
            trade_date varchar not null,
            account_id varchar not null,
            strategy_id varchar not null,
            code varchar not null,
            target_weight double,
            target_value double,
            source_bundle_id varchar,
            source_sleeve varchar,
            score_version varchar,
            reason varchar,
            generated_at varchar,
            primary key (trade_date, account_id, strategy_id, code)
        )
        """,
        """
        create table if not exists live_orders (
            order_id varchar primary key,
            trade_date varchar not null,
            account_id varchar not null,
            strategy_id varchar not null,
            code varchar not null,
            side varchar not null,
            order_qty double,
            order_price double,
            status varchar,
            block_reason varchar,
            created_at varchar,
            submitted_at varchar,
            updated_at varchar
        )
        """,
        """
        create table if not exists live_fills (
            fill_id varchar primary key,
            order_id varchar,
            trade_date varchar,
            code varchar,
            side varchar,
            fill_qty double,
            fill_price double,
            fill_time varchar,
            commission double,
            tax double,
            slippage_bps double
        )
        """,
        """
        create table if not exists live_risk_logs (
            trade_date varchar,
            account_id varchar,
            strategy_id varchar,
            check_name varchar,
            severity varchar,
            passed boolean,
            action varchar,
            reason varchar,
            created_at varchar
        )
        """,
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
    _migrate_backtest_output_tables(con)
    _migrate_portfolio_targets_table(con)
    _migrate_backtest_metrics_primary_key(con)


def _migrate_backtest_output_tables(con: duckdb.DuckDBPyConnection) -> None:
    specs = {
        "ml_backtest_orders": """
            run_id varchar,
            fold_id varchar,
            strategy_id varchar,
            score_version varchar,
            sim_date varchar,
            decision_date varchar,
            code varchar,
            side varchar,
            order_seq integer,
            qty double,
            target_weight double,
            order_px_ref varchar,
            fill_px double,
            status varchar,
            reason varchar,
            entry_date varchar,
            entry_price double,
            exit_date varchar,
            holding_days integer,
            entry_trade_score double,
            exit_trade_score double,
            entry_abs_rank_pct double,
            entry_risk_rank_pct double,
            entry_reason varchar,
            exit_reason varchar,
            sell_blocked_reason varchar,
            realized_pnl double
        """,
        "ml_backtest_positions": """
            run_id varchar,
            fold_id varchar,
            strategy_id varchar,
            score_version varchar,
            sim_date varchar,
            code varchar,
            position_qty double,
            market_value double,
            weight double,
            entry_date varchar,
            entry_price double,
            holding_days integer,
            entry_trade_score double,
            entry_reason varchar
        """,
        "ml_backtest_nav": """
            run_id varchar,
            fold_id varchar,
            strategy_id varchar,
            score_version varchar,
            sim_date varchar,
            nav double,
            cash double,
            gross_exposure double,
            turnover double
        """,
    }
    for table_name, columns_sql in specs.items():
        row = con.execute(
            "select sql from duckdb_tables() where table_name = ?",
            [table_name],
        ).fetchone()
        if row is None:
            continue
        table_sql = str(row[0]).lower().replace(" ", "")
        if "strategy_id" in table_sql and "score_version" in table_sql and "primarykey(" not in table_sql:
            continue
        temp_name = f"_{table_name}_migrate_{uuid4().hex}"
        con.execute(f"create table {temp_name} ({columns_sql})")
        con.execute(f"insert into {temp_name} by name select * from {table_name}")
        con.execute(f"drop table {table_name}")
        con.execute(f"alter table {temp_name} rename to {table_name}")


def _migrate_portfolio_targets_table(con: duckdb.DuckDBPyConnection) -> None:
    row = con.execute(
        "select sql from duckdb_tables() where table_name = 'ml_portfolio_targets_daily'"
    ).fetchone()
    if row is None:
        return
    table_sql = str(row[0]).lower().replace(" ", "")
    expected_pk = "primarykey(trade_date,run_id,fold_id,portfolio_id,score_version,code)"
    if expected_pk in table_sql:
        return
    temp_name = f"_ml_portfolio_targets_daily_migrate_{uuid4().hex}"
    con.execute(
        f"""
        create table {temp_name} (
            run_id varchar,
            fold_id varchar,
            trade_date varchar not null,
            portfolio_id varchar not null,
            score_version varchar,
            code varchar not null,
            target_weight double,
            rank_n integer,
            trade_score double,
            entry_reason varchar,
            signal_action varchar,
            hold_reason varchar,
            exit_reason varchar,
            sell_blocked_reason varchar,
            entry_date varchar,
            entry_price double,
            shares double,
            holding_days integer,
            entry_trade_score double,
            latest_trade_score double,
            generated_at varchar,
            primary key (trade_date, run_id, fold_id, portfolio_id, score_version, code)
        )
        """
    )
    con.execute(
        f"""
        insert into {temp_name} by name
        select
            * replace (
                coalesce(run_id, 'legacy') as run_id,
                coalesce(fold_id, portfolio_id, 'legacy') as fold_id,
                coalesce(score_version, 'legacy') as score_version
            )
        from ml_portfolio_targets_daily
        """
    )
    con.execute("drop table ml_portfolio_targets_daily")
    con.execute(f"alter table {temp_name} rename to ml_portfolio_targets_daily")


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
    frame = _normalize_frame_for_table(table_name, frame)
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


def _normalize_frame_for_table(table_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if table_name != "ml_portfolio_targets_daily":
        return frame
    out = frame.copy()
    if "run_id" not in out:
        out["run_id"] = "legacy"
    if "fold_id" not in out:
        out["fold_id"] = out["portfolio_id"].astype(str) if "portfolio_id" in out else "legacy"
    if "score_version" not in out:
        out["score_version"] = "legacy"
    return out


def clear_backtest_outputs(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    fold_id: str,
    strategy_id: str,
    score_version: str,
    start_date: str,
    end_date: str,
) -> None:
    con.execute(
        """
        delete from ml_backtest_orders
        where run_id = ?
          and fold_id = ?
          and strategy_id = ?
          and score_version = ?
          and decision_date between ? and ?
        """,
        [run_id, fold_id, strategy_id, score_version, start_date, end_date],
    )
    con.execute(
        """
        delete from ml_backtest_positions
        where run_id = ?
          and fold_id = ?
          and strategy_id = ?
          and score_version = ?
          and sim_date between ? and ?
        """,
        [run_id, fold_id, strategy_id, score_version, start_date, end_date],
    )
    con.execute(
        """
        delete from ml_backtest_nav
        where run_id = ?
          and fold_id = ?
          and strategy_id = ?
          and score_version = ?
          and sim_date between ? and ?
        """,
        [run_id, fold_id, strategy_id, score_version, start_date, end_date],
    )


def clear_portfolio_targets(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    fold_id: str,
    portfolio_id: str,
    score_version: str,
    start_date: str,
    end_date: str,
) -> None:
    con.execute(
        """
        delete from ml_portfolio_targets_daily
        where run_id = ?
          and fold_id = ?
          and portfolio_id = ?
          and score_version = ?
          and trade_date between ? and ?
        """,
        [run_id, fold_id, portfolio_id, score_version, start_date, end_date],
    )
