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
from ml_stock_selector.serving.live_sim import LiveSimConfig, init_live_sim_db, run_live_sim_day
from ml_stock_selector.serving.qmt_order_export import DEFAULT_QMT_ORDER_JSON, export_qmt_orders
from ml_stock_selector.storage import init_ml_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_walkforward_adv10m.toml")
    parser.add_argument("--as-of-date")
    parser.add_argument("--state-db", default="outputs/ml/live_sim/live_sim_state.duckdb")
    parser.add_argument("--report-dir", default="outputs/ml/live_sim/reports")
    parser.add_argument("--account-id", default="preferred_adv10m_paper")
    parser.add_argument("--initial-cash", type=float, default=300_000.0)
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
        as_of_date = args.as_of_date or _latest_feature_date(ml_con, str(config.features["feature_set_id"]))
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
        bars = load_live_unadjusted_stock_bars(
            str(config.data["alpha_data_db"]),
            as_of_date,
            as_of_date,
            str(config.data["normalized_bars_table"]),
        )
        result = run_live_sim_day(
            live_con,
            as_of_date,
            predictions,
            bars,
            LiveSimConfig(
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
