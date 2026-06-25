from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_stock_selector.config import load_ml_config
from ml_stock_selector.data_access import load_live_unadjusted_stock_bars
from ml_stock_selector.feature_store_reader import FeatureStoreSpec
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.holding_policy import HoldingPolicy
from ml_stock_selector.serving.daily_signal import generate_daily_signal
from ml_stock_selector.serving.live_sim import (
    PROFIT_PROTECT_RUN_ID,
    PROFIT_PROTECT_PORTFOLIO_ID,
    PROFIT_PROTECT_SCORE_VERSION,
    init_live_sim_db,
    load_active_live_model_bundle,
    load_live_predictions,
    profit_protect_live_sim_config,
    run_live_sim_day,
)
from ml_stock_selector.serving.qmt_order_export import DEFAULT_QMT_ORDER_JSON, export_qmt_orders
from ml_stock_selector.storage import init_ml_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_walkforward_adv10m_ret5_alpha_risk.toml")
    parser.add_argument("--as-of-date")
    parser.add_argument("--state-db", default="outputs/ml/live_sim/live_sim_state.duckdb")
    parser.add_argument("--report-dir", default="outputs/ml/live_sim/reports")
    parser.add_argument("--account-id", default="profit_protect_paper")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--prediction-run-id", default=PROFIT_PROTECT_RUN_ID)
    parser.add_argument("--prediction-fold-id")
    parser.add_argument("--prediction-score-version", default=PROFIT_PROTECT_SCORE_VERSION)
    parser.add_argument("--prediction-source", choices=["auto", "live", "shared"], default="auto")
    parser.add_argument("--live-bundle-id")
    parser.add_argument("--generate-daily-signal", action="store_true")
    parser.add_argument("--feature-store-dir", default="outputs/ml/feature_store")
    parser.add_argument("--feature-store-version", default="v2_pv_only_001")
    parser.add_argument("--use-feature-store", type=_parse_bool, default=True)
    parser.add_argument("--qmt-order-json", default=str(DEFAULT_QMT_ORDER_JSON))
    parser.add_argument("--qmt-order-copy-dir")
    args = parser.parse_args()

    config = load_ml_config(args.config)
    ml_con = init_ml_db(str(config.data["ml_db"]))
    live_con = init_live_sim_db(args.state_db)
    try:
        as_of_date = args.as_of_date or (
            _latest_feature_date(ml_con, str(config.features["feature_set_id"]))
            if args.generate_daily_signal
            else _latest_prediction_date(
                ml_con,
                args.prediction_run_id,
                args.prediction_score_version,
            )
        )
        if args.generate_daily_signal:
            predictions, _targets = generate_daily_signal(
                ml_con,
                as_of_date,
                str(config.features["feature_set_id"]),
                int(config.labels["main_horizon"]),
                "live_sim_staging",
                constraints=_loose_prediction_constraints(),
                use_v2=True,
                exclude_bse=bool(config.universe.get("exclude_bse", True)),
                feature_store_spec=FeatureStoreSpec(
                    args.feature_store_dir,
                    args.feature_store_version,
                    str(config.features["feature_set_id"]),
                )
                if args.use_feature_store
                else None,
            )
        else:
            predictions = _load_predictions_for_live_sim(
                live_con,
                ml_con,
                as_of_date,
                args.prediction_source,
                args.live_bundle_id,
                args.prediction_run_id,
                args.prediction_fold_id or f"wf_{as_of_date[:4]}",
                args.prediction_score_version,
            )
        bars = load_live_unadjusted_stock_bars(
            str(config.data["alpha_data_db"]),
            _lookback_start(as_of_date),
            as_of_date,
            str(config.data["normalized_bars_table"]),
        )
        result = run_live_sim_day(
            live_con,
            as_of_date,
            predictions,
            bars,
            profit_protect_live_sim_config(
                account_id=args.account_id,
                initial_cash=args.initial_cash,
                report_dir=Path(args.report_dir),
            ),
        )
        qmt_order_json = export_qmt_orders(
            live_con,
            account_id=args.account_id,
            decision_date=result.plan_date,
            execution_date=result.execution_date,
            output_json=args.qmt_order_json,
            copy_dir=args.qmt_order_copy_dir,
        )
    finally:
        ml_con.close()
        live_con.close()
    print(
        " ".join(
            [
                f"date={result.plan_date}",
                f"execution_date={result.execution_date}",
                f"planned={len(result.planned_orders)}",
                f"executions={len(result.executions)}",
                f"nav={float(result.nav['nav']):.2f}",
                f"report={result.report_path}",
                f"qmt_order_json={qmt_order_json}",
            ]
        )
    )


def _latest_prediction_date(con, run_id: str, score_version: str) -> str:
    row = con.execute(
        """
        select max(trade_date)
        from ml_predictions_daily
        where run_id = ?
          and score_version = ?
        """,
        [run_id, score_version],
    ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError(f"no predictions found for run_id={run_id}, score_version={score_version}")
    return str(row[0])[:10]


def _load_existing_predictions(
    con,
    as_of_date: str,
    run_id: str,
    fold_id: str,
    score_version: str,
) -> pd.DataFrame:
    frame = con.execute(
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
        where p.trade_date = ?
          and p.run_id = ?
          and p.fold_id = ?
          and p.score_version = ?
        order by p.code
        """,
        [as_of_date, run_id, fold_id, score_version],
    ).fetchdf()
    if frame.empty:
        raise RuntimeError(
            f"no predictions found for date={as_of_date}, run_id={run_id}, fold_id={fold_id}, score_version={score_version}"
        )
    return frame.drop_duplicates(["trade_date", "code"], keep="last").reset_index(drop=True)


def _load_predictions_for_live_sim(
    live_con,
    ml_con,
    as_of_date: str,
    prediction_source: str,
    live_bundle_id: str | None,
    run_id: str,
    fold_id: str,
    score_version: str,
) -> pd.DataFrame:
    if prediction_source in {"auto", "live"}:
        try:
            bundle_id = live_bundle_id or str(load_active_live_model_bundle(live_con, PROFIT_PROTECT_PORTFOLIO_ID)["bundle_id"])
            live_predictions = _load_live_predictions_with_tradeability(live_con, ml_con, as_of_date, bundle_id)
            if not live_predictions.empty:
                return live_predictions
            if prediction_source == "live":
                raise RuntimeError(f"no live predictions found for date={as_of_date}, bundle_id={bundle_id}")
        except RuntimeError:
            if prediction_source == "live":
                raise
    return _load_existing_predictions(ml_con, as_of_date, run_id, fold_id, score_version)


def _load_live_predictions_with_tradeability(
    live_con,
    ml_con,
    as_of_date: str,
    bundle_id: str,
) -> pd.DataFrame:
    live_predictions = load_live_predictions(live_con, as_of_date, bundle_id=bundle_id)
    if live_predictions.empty:
        return live_predictions
    tradeability = ml_con.execute(
        """
        select
            trade_date,
            code,
            industry_code,
            industry_name,
            is_st,
            is_paused,
            is_bse,
            adv20_amount,
            can_buy_next_open,
            can_sell_next_open,
            next_open,
            next_limit_up,
            next_limit_down,
            next_is_paused,
            limit_up_pct,
            limit_down_pct,
            limit_band
        from ml_tradeability_daily
        where trade_date = ?
        """,
        [as_of_date],
    ).fetchdf()
    merged = live_predictions.merge(
        tradeability,
        on=["trade_date", "code"],
        how="left",
        suffixes=("", "_tradeability"),
    )
    if "adv20_amount_tradeability" in merged:
        merged["adv20_amount"] = merged["adv20_amount_tradeability"].fillna(merged["adv20_amount"])
        merged = merged.drop(columns=["adv20_amount_tradeability"])
    return merged.drop_duplicates(["trade_date", "code"], keep="last").reset_index(drop=True)


def _lookback_start(as_of_date: str, calendar_days: int = 14) -> str:
    return (pd.Timestamp(as_of_date) - pd.Timedelta(days=calendar_days)).date().isoformat()


def _latest_feature_date(con, feature_set_id: str) -> str:
    row = con.execute(
        "select max(trade_date) from ml_feature_mart_daily where feature_set_id = ?",
        [feature_set_id],
    ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError(f"no feature rows found for feature_set_id={feature_set_id}")
    return str(row[0])[:10]


def _loose_prediction_constraints() -> PortfolioConstraints:
    return PortfolioConstraints(
        target_positions=12,
        hard_max_positions=12,
        max_initial_entries=12,
        max_new_entries_per_day=12,
        min_adv20_amount=0.0,
        candidate_min_trade_score=0.0,
        core_min_trade_score=0.0,
        candidate_absolute_min_rank_pct=0.0,
        candidate_active_min_rank_pct=0.0,
        candidate_risk_max_rank_pct=1.0,
        core_absolute_min_rank_pct=0.0,
        core_active_min_rank_pct=0.0,
        core_risk_max_rank_pct=1.0,
        holding_policy=HoldingPolicy(sell_score_threshold=-1.0),
    )


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


if __name__ == "__main__":
    main()
