from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.backtest.data_access import load_backtest_candidates
from ml_stock_selector.backtest.engine import BacktestResult
from ml_stock_selector.backtest.metrics import summarize_fold_metric_rows, summarize_walkforward_metric_rows
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.data_access import load_normalized_stock_bars
from ml_stock_selector.runtime.artifacts import write_backtest_fold_artifacts
from ml_stock_selector.runtime.run_context import create_run_context, register_run_context, register_run_fold, update_run_status
from ml_stock_selector.serving.live_sim import LiveSimConfig, init_live_sim_db, run_live_sim_day
from ml_stock_selector.storage import clear_backtest_outputs, clear_portfolio_targets, init_ml_db, upsert_dataframe


DEFAULT_STRATEGY_ID = "live_sim_replay_v1"
DEFAULT_SCORE_VERSION = "preferred_adv10m_fulladv015_top12_live_replay"
DEFAULT_PREDICTION_SCORE_VERSION = "v2_three_model"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_walkforward_adv10m.toml")
    parser.add_argument("--ml-db")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fold-id", action="append")
    parser.add_argument("--prediction-run-id", required=True)
    parser.add_argument("--prediction-fold-id")
    parser.add_argument("--prediction-score-version", default=DEFAULT_PREDICTION_SCORE_VERSION)
    parser.add_argument("--score-version", default=DEFAULT_SCORE_VERSION)
    parser.add_argument("--strategy-id", default=DEFAULT_STRATEGY_ID)
    parser.add_argument("--run-artifact-dir", default="outputs/ml/runs")
    parser.add_argument("--state-dir")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_ml_config(args.config)
    ml_db = args.ml_db or str(config.data["ml_db"])
    selected_folds = _select_folds(config.split.get("folds", []), args.fold_id)
    if not selected_folds:
        raise ValueError("No folds selected")

    con = init_ml_db(ml_db)
    context = create_run_context(
        run_type="live_sim_replay",
        run_id=args.run_id,
        experiment_name="live_sim_replay",
        config_path=args.config,
        artifact_root=args.run_artifact_dir,
        alpha_data_db=str(config.data["alpha_data_db"]),
        ml_db=ml_db,
        feature_set_id=str(config.features["feature_set_id"]),
        label_version=f"{config.labels['label_base']}_h{config.labels['main_horizon']}",
        score_version=args.score_version,
    )
    register_run_context(con, context)
    fold_metric_frames: list[pd.DataFrame] = []
    try:
        for fold in selected_folds:
            fold_id = str(fold["fold_id"])
            register_run_fold(con, context, fold, status="running")
            result = replay_fold(con, config, context.artifact_root, args, fold)
            fold_metric_frames.append(result["metrics"])
            register_run_fold(con, context, fold, status="success")
        if fold_metric_frames:
            summary = summarize_walkforward_metric_rows(
                pd.concat(fold_metric_frames, ignore_index=True),
                run_id=args.run_id,
                score_version=args.score_version,
                strategy_id=args.strategy_id,
            )
            upsert_dataframe(con, "ml_backtest_metrics", summary, ["run_id", "fold_id", "score_version", "metric_name", "segment"])
        update_run_status(con, context, "success")
    finally:
        con.close()
    print(f"folds={len(selected_folds)}")


def replay_fold(con, config, artifact_root: Path, args, fold: dict[str, object]) -> dict[str, pd.DataFrame]:
    fold_id = str(fold["fold_id"])
    prediction_fold_id = args.prediction_fold_id or fold_id
    predictions = load_backtest_candidates(
        con,
        run_id=args.prediction_run_id,
        fold_id=prediction_fold_id,
        score_version=args.prediction_score_version,
        exclude_bse=bool(config.universe.get("exclude_bse", False)),
    )
    if predictions.empty:
        raise ValueError(f"No predictions for run={args.prediction_run_id} fold={prediction_fold_id}")
    start_date = str(predictions["trade_date"].min())[:10]
    end_date = str(predictions["trade_date"].max())[:10]
    bars = load_normalized_stock_bars(
        str(config.data["alpha_data_db"]),
        start_date,
        end_date,
        str(config.data["normalized_bars_table"]),
    )
    state_path = _state_path(artifact_root, args, fold_id)
    if state_path.exists():
        state_path.unlink()
    live_con = init_live_sim_db(state_path)
    live_config = replace(
        LiveSimConfig(),
        account_id=f"{args.run_id}_{fold_id}",
        report_dir=artifact_root / "live_reports" / fold_id,
    )
    orders_by_day: list[pd.DataFrame] = []
    positions_by_day: list[pd.DataFrame] = []
    targets_by_day: list[pd.DataFrame] = []
    try:
        for trade_date, day_predictions in predictions.groupby("trade_date", sort=True):
            date_key = str(trade_date)[:10]
            day_result = run_live_sim_day(live_con, date_key, day_predictions.copy(), bars, live_config)
            if not day_result.executions.empty:
                orders_by_day.append(_orders_for_storage(day_result.executions, args, fold_id))
            if not day_result.holdings.empty:
                positions_by_day.append(_positions_for_storage(day_result.holdings, bars, date_key, day_result.nav, args, fold_id))
            if not day_result.planned_orders.empty:
                targets_by_day.append(_targets_for_storage(day_result.planned_orders, args, fold_id, live_config.portfolio_id))
        nav = _nav_for_storage(live_con, args, fold_id)
    finally:
        live_con.close()

    orders = pd.concat(orders_by_day, ignore_index=True) if orders_by_day else _empty_orders()
    positions = pd.concat(positions_by_day, ignore_index=True) if positions_by_day else _empty_positions()
    targets = pd.concat(targets_by_day, ignore_index=True) if targets_by_day else _empty_targets()
    diagnostics = pd.DataFrame()
    result = BacktestResult(orders, positions, nav, diagnostics)
    metrics = summarize_fold_metric_rows(
        result,
        run_id=args.run_id,
        fold_id=fold_id,
        score_version=args.score_version,
        strategy_id=args.strategy_id,
        start_date=start_date,
        end_date=end_date,
        candidate_pool_size=0.0,
        core_pool_size=0.0,
        bse_excluded_count=0.0,
    )
    clear_backtest_outputs(con, args.run_id, fold_id, args.strategy_id, args.score_version, start_date, end_date)
    clear_portfolio_targets(con, args.run_id, fold_id, live_config.portfolio_id, args.score_version, start_date, end_date)
    upsert_dataframe(con, "ml_backtest_nav", nav, ["run_id", "fold_id", "strategy_id", "score_version", "sim_date"])
    if not orders.empty:
        upsert_dataframe(con, "ml_backtest_orders", orders, ["run_id", "fold_id", "strategy_id", "score_version", "sim_date", "decision_date", "code", "side", "order_seq"])
    if not positions.empty:
        upsert_dataframe(con, "ml_backtest_positions", positions, ["run_id", "fold_id", "strategy_id", "score_version", "sim_date", "code"])
    if not targets.empty:
        upsert_dataframe(con, "ml_portfolio_targets_daily", targets, ["trade_date", "run_id", "fold_id", "portfolio_id", "score_version", "code"])
    upsert_dataframe(con, "ml_backtest_metrics", metrics, ["run_id", "fold_id", "score_version", "metric_name", "segment"])
    write_backtest_fold_artifacts(
        artifact_root,
        fold_id=fold_id,
        strategy_id=args.strategy_id,
        score_version=args.score_version,
        portfolio_id=live_config.portfolio_id,
        backtest_params={
            "cli_args": vars(args),
            "live_sim": live_config,
            "prediction_run_id": args.prediction_run_id,
            "prediction_fold_id": prediction_fold_id,
            "prediction_score_version": args.prediction_score_version,
            "start_date": start_date,
            "end_date": end_date,
        },
        targets=targets,
        diagnostics=diagnostics,
        orders=orders,
        positions=positions,
        nav=nav,
        metrics=metrics,
    )
    return {"metrics": metrics, "orders": orders, "positions": positions, "nav": nav, "targets": targets}


def _select_folds(folds: list[dict[str, object]], selected: list[str] | None) -> list[dict[str, object]]:
    if selected is None:
        return list(folds)
    wanted = set(selected)
    out = [fold for fold in folds if str(fold.get("fold_id")) in wanted]
    missing = wanted - {str(fold.get("fold_id")) for fold in out}
    if missing:
        raise ValueError(f"Unknown fold_id: {sorted(missing)}")
    return out


def _state_path(artifact_root: Path, args, fold_id: str) -> Path:
    state_dir = Path(args.state_dir) if args.state_dir else artifact_root / "live_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{fold_id}.duckdb"


def _orders_for_storage(executions: pd.DataFrame, args, fold_id: str) -> pd.DataFrame:
    out = executions.copy()
    out["run_id"] = args.run_id
    out["fold_id"] = fold_id
    out["strategy_id"] = args.strategy_id
    out["score_version"] = args.score_version
    out["order_seq"] = range(1, len(out) + 1)
    return out


def _positions_for_storage(holdings: pd.DataFrame, bars: pd.DataFrame, date_key: str, nav: dict[str, object], args, fold_id: str) -> pd.DataFrame:
    prices = _latest_close_prices(bars, date_key)
    rows = []
    nav_value = float(nav.get("nav") or 0.0)
    for row in holdings.itertuples(index=False):
        code = str(row.code)
        qty = float(row.qty)
        price = float(prices.get(code, 0.0))
        market_value = qty * price
        rows.append(
            {
                "run_id": args.run_id,
                "fold_id": fold_id,
                "strategy_id": args.strategy_id,
                "score_version": args.score_version,
                "sim_date": date_key,
                "code": code,
                "position_qty": qty,
                "market_value": market_value,
                "weight": market_value / nav_value if nav_value else 0.0,
                "entry_date": getattr(row, "entry_date", None),
                "entry_price": getattr(row, "entry_price", None),
                "holding_days": _calendar_days(getattr(row, "entry_date", None), date_key),
                "entry_trade_score": getattr(row, "entry_trade_score", None),
                "entry_reason": getattr(row, "entry_reason", None),
            }
        )
    return pd.DataFrame(rows)


def _targets_for_storage(planned: pd.DataFrame, args, fold_id: str, portfolio_id: str) -> pd.DataFrame:
    out = planned.rename(columns={"decision_date": "trade_date", "trade_score_v2": "trade_score"}).copy()
    out["run_id"] = args.run_id
    out["fold_id"] = fold_id
    out["portfolio_id"] = portfolio_id
    out["score_version"] = args.score_version
    out["rank_n"] = range(1, len(out) + 1)
    out["generated_at"] = out.get("generated_at")
    keep = [
        "run_id",
        "fold_id",
        "trade_date",
        "portfolio_id",
        "score_version",
        "code",
        "target_weight",
        "rank_n",
        "trade_score",
        "entry_reason",
        "signal_action",
        "generated_at",
    ]
    return out[[column for column in keep if column in out.columns]]


def _nav_for_storage(live_con, args, fold_id: str) -> pd.DataFrame:
    nav = live_con.execute(
        """
        select sim_date, nav, cash, holding_market_value
        from live_sim_nav
        where sim_date <> 'INITIAL'
        order by sim_date
        """
    ).fetchdf()
    if nav.empty:
        return pd.DataFrame(columns=["run_id", "fold_id", "strategy_id", "score_version", "sim_date", "nav", "cash", "gross_exposure", "turnover"])
    executions = live_con.execute(
        """
        select sim_date, sum(case when status = 'filled' then qty * fill_px else 0 end) as traded_value
        from live_sim_executions
        group by 1
        """
    ).fetchdf()
    turnover = executions.set_index("sim_date")["traded_value"].to_dict() if not executions.empty else {}
    nav["run_id"] = args.run_id
    nav["fold_id"] = fold_id
    nav["strategy_id"] = args.strategy_id
    nav["score_version"] = args.score_version
    nav["gross_exposure"] = nav["holding_market_value"] / nav["nav"].replace(0.0, pd.NA)
    nav["turnover"] = [float(turnover.get(date, 0.0) or 0.0) / float(value) if value else 0.0 for date, value in zip(nav["sim_date"], nav["nav"])]
    return nav[["run_id", "fold_id", "strategy_id", "score_version", "sim_date", "nav", "cash", "gross_exposure", "turnover"]]


def _latest_close_prices(bars: pd.DataFrame, as_of_date: str) -> dict[str, float]:
    frame = bars[bars["trade_date"].astype(str) <= as_of_date].sort_values(["code", "trade_date"])
    if frame.empty:
        return {}
    return {str(row.code): float(row.close) for row in frame.drop_duplicates("code", keep="last").itertuples(index=False)}


def _calendar_days(entry_date: object, as_of_date: str) -> int | None:
    start = pd.to_datetime(entry_date, errors="coerce")
    end = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return None
    return max(int((end - start).days), 0)


def _empty_orders() -> pd.DataFrame:
    return pd.DataFrame(columns=["run_id", "fold_id", "strategy_id", "score_version", "sim_date", "decision_date", "code", "side", "order_seq"])


def _empty_positions() -> pd.DataFrame:
    return pd.DataFrame(columns=["run_id", "fold_id", "strategy_id", "score_version", "sim_date", "code"])


def _empty_targets() -> pd.DataFrame:
    return pd.DataFrame(columns=["run_id", "fold_id", "trade_date", "portfolio_id", "score_version", "code"])


if __name__ == "__main__":
    main()
