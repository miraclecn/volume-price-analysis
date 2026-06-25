from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_stock_selector.config import load_ml_config
from ml_stock_selector.data_access import load_live_unadjusted_stock_bars
from ml_stock_selector.serving.live_sim import (
    PROFIT_PROTECT_RUN_ID,
    PROFIT_PROTECT_SCORE_VERSION,
    init_live_sim_db,
    profit_protect_live_sim_config,
    run_live_sim_day,
)
from ml_stock_selector.serving.qmt_order_export import DEFAULT_QMT_ORDER_JSON, export_qmt_orders
from ml_stock_selector.storage import init_ml_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_walkforward_adv10m_ret5_alpha_risk.toml")
    parser.add_argument("--state-db", default="outputs/ml/live_sim/live_sim_state.duckdb")
    parser.add_argument("--report-dir", default="outputs/ml/live_sim/reports")
    parser.add_argument("--qmt-order-json", default=str(DEFAULT_QMT_ORDER_JSON))
    parser.add_argument("--account-id", default="profit_protect_paper")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--replay-start-date", default="2026-01-05")
    parser.add_argument("--activation-date", default="2026-06-16")
    parser.add_argument("--end-date")
    parser.add_argument("--prediction-run-id", default=PROFIT_PROTECT_RUN_ID)
    parser.add_argument("--prediction-fold-id", default="wf_2026")
    parser.add_argument("--prediction-score-version", default=PROFIT_PROTECT_SCORE_VERSION)
    parser.add_argument("--backup-dir", default="outputs/ml/live_sim/archive")
    parser.add_argument("--temp-state-db", default="/tmp/vpa_profit_protect_activation_replay.duckdb")
    args = parser.parse_args()

    config = load_ml_config(args.config)
    ml_con = init_ml_db(str(config.data["ml_db"]))
    try:
        end_date = args.end_date or latest_prediction_date(
            ml_con,
            args.prediction_run_id,
            args.prediction_fold_id,
            args.prediction_score_version,
        )
        predictions = load_predictions(
            ml_con,
            args.replay_start_date,
            end_date,
            args.prediction_run_id,
            args.prediction_fold_id,
            args.prediction_score_version,
        )
    finally:
        ml_con.close()

    if predictions.empty:
        raise RuntimeError("no predictions loaded")
    if args.activation_date not in set(predictions["trade_date"].astype(str).str[:10]):
        raise RuntimeError(f"activation date has no predictions: {args.activation_date}")

    bars = load_live_unadjusted_stock_bars(
        str(config.data["alpha_data_db"]),
        lookback_start(args.replay_start_date),
        end_date,
        str(config.data["normalized_bars_table"]),
    )
    replay_buys = continuous_activation_buys(
        predictions[predictions["trade_date"].astype(str).str[:10] <= args.activation_date].copy(),
        bars,
        args.temp_state_db,
        args.account_id,
        args.initial_cash,
        Path(args.report_dir) / "_activation_replay_reference",
        args.activation_date,
    )
    if not replay_buys:
        raise RuntimeError(f"continuous replay produced no activation buys for {args.activation_date}")

    state_path = Path(args.state_db)
    backup_state_db(state_path, Path(args.backup_dir))
    clear_live_outputs(Path(args.report_dir), Path(args.qmt_order_json))
    live_con = init_live_sim_db(state_path)
    live_config = profit_protect_live_sim_config(
        account_id=args.account_id,
        initial_cash=args.initial_cash,
        report_dir=Path(args.report_dir),
    )
    try:
        latest_result = None
        for trade_date, day_predictions in predictions[predictions["trade_date"].astype(str).str[:10] >= args.activation_date].groupby("trade_date", sort=True):
            date_key = str(trade_date)[:10]
            latest_result = run_live_sim_day(live_con, date_key, day_predictions.copy(), bars, live_config)
            if date_key == args.activation_date:
                prune_activation_plan(live_con, args.account_id, args.activation_date, replay_buys)
                latest_result = run_live_sim_day(live_con, date_key, day_predictions.copy(), bars, live_config)
        if latest_result is None:
            raise RuntimeError("no live dates were replayed")
        qmt_order_json = export_qmt_orders(
            live_con,
            account_id=args.account_id,
            decision_date=latest_result.plan_date,
            execution_date=latest_result.execution_date,
            output_json=args.qmt_order_json,
        )
        current_holdings = live_con.execute(
            "select count(*) from live_sim_holdings where account_id = ?",
            [args.account_id],
        ).fetchone()[0]
    finally:
        live_con.close()

    print(
        " ".join(
            [
                f"activation_date={args.activation_date}",
                f"activation_buys={','.join(replay_buys)}",
                f"end_date={latest_result.plan_date}",
                f"execution_date={latest_result.execution_date}",
                f"planned={len(latest_result.planned_orders)}",
                f"executions={len(latest_result.executions)}",
                f"holdings={current_holdings}",
                f"nav={float(latest_result.nav['nav']):.2f}",
                f"qmt_order_json={qmt_order_json}",
            ]
        )
    )


def latest_prediction_date(con, run_id: str, fold_id: str, score_version: str) -> str:
    row = con.execute(
        """
        select max(trade_date)
        from ml_predictions_daily
        where run_id = ?
          and fold_id = ?
          and score_version = ?
        """,
        [run_id, fold_id, score_version],
    ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError(f"no predictions found for run_id={run_id}, fold_id={fold_id}, score_version={score_version}")
    return str(row[0])[:10]


def load_predictions(
    con,
    start_date: str,
    end_date: str,
    run_id: str,
    fold_id: str,
    score_version: str,
) -> pd.DataFrame:
    return con.execute(
        """
        select
            p.*,
            t.industry_code,
            t.industry_name,
            t.is_st,
            t.is_paused,
            t.is_bse,
            t.adv20_amount,
            t.can_buy_next_open,
            t.can_sell_next_open,
            t.next_open,
            t.next_limit_up,
            t.next_limit_down,
            t.next_is_paused,
            t.limit_up_pct,
            t.limit_down_pct,
            t.limit_band
        from ml_predictions_daily p
        left join ml_tradeability_daily t
          on p.trade_date = t.trade_date and p.code = t.code
        where p.run_id = ?
          and p.fold_id = ?
          and p.score_version = ?
          and p.trade_date between ? and ?
        order by p.trade_date, p.code
        """,
        [run_id, fold_id, score_version, start_date, end_date],
    ).fetchdf()


def continuous_activation_buys(
    predictions: pd.DataFrame,
    bars: pd.DataFrame,
    temp_state_db: str,
    account_id: str,
    initial_cash: float,
    report_dir: Path,
    activation_date: str,
) -> list[str]:
    temp_path = Path(temp_state_db)
    temp_path.unlink(missing_ok=True)
    shutil.rmtree(report_dir, ignore_errors=True)
    con = init_live_sim_db(temp_path)
    config = profit_protect_live_sim_config(
        account_id=f"{account_id}_continuous_reference",
        initial_cash=initial_cash,
        report_dir=report_dir,
    )
    try:
        for trade_date, day_predictions in predictions.groupby("trade_date", sort=True):
            run_live_sim_day(con, str(trade_date)[:10], day_predictions.copy(), bars, config)
        plan = con.execute(
            """
            select code, side
            from live_sim_planned_orders
            where account_id = ?
              and decision_date = ?
            order by code
            """,
            [config.account_id, activation_date],
        ).fetchdf()
    finally:
        con.close()
    return activation_buy_codes(plan)


def activation_buy_codes(plan: pd.DataFrame) -> list[str]:
    if plan.empty:
        return []
    return sorted(plan.loc[plan["side"].astype(str).str.lower() == "buy", "code"].astype(str).unique().tolist())


def prune_activation_plan(
    con: duckdb.DuckDBPyConnection,
    account_id: str,
    activation_date: str,
    allowed_buy_codes: list[str],
) -> None:
    existing = con.execute(
        """
        select code, side
        from live_sim_planned_orders
        where account_id = ?
          and decision_date = ?
        """,
        [account_id, activation_date],
    ).fetchdf()
    planned_buys = activation_buy_codes(existing)
    missing = sorted(set(allowed_buy_codes) - set(planned_buys))
    if missing:
        raise RuntimeError(f"activation buys not present in cold-start plan: {missing}")

    allowed = pd.DataFrame({"code": allowed_buy_codes})
    con.register("_activation_allowed_buys", allowed)
    try:
        con.execute(
            """
            delete from live_sim_planned_orders
            where account_id = ?
              and decision_date = ?
              and lower(side) = 'buy'
              and code not in (select code from _activation_allowed_buys)
            """,
            [account_id, activation_date],
        )
        con.execute(
            """
            delete from live_sim_planned_orders
            where account_id = ?
              and decision_date = ?
              and lower(side) <> 'buy'
            """,
            [account_id, activation_date],
        )
    finally:
        con.unregister("_activation_allowed_buys")


def backup_state_db(state_path: Path, backup_dir: Path) -> None:
    if not state_path.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    shutil.copy2(state_path, backup_dir / f"live_sim_state_before_activation_reset_{stamp}.duckdb")
    state_path.unlink()


def clear_live_outputs(report_dir: Path, qmt_order_json: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    for path in report_dir.glob("live_sim_summary_*.md"):
        path.unlink()
    qmt_order_json.parent.mkdir(parents=True, exist_ok=True)
    for path in qmt_order_json.parent.glob("*.json"):
        path.unlink()


def lookback_start(start_date: str) -> str:
    return (datetime.fromisoformat(start_date) - timedelta(days=21)).date().isoformat()


if __name__ == "__main__":
    main()
