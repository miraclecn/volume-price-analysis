from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.serving.live import (
    build_live_orders,
    build_live_target_positions,
    build_risk_logs,
    record_live_fills,
)
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--ml-db")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--execution-date")
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--account-nav", type=float, required=True)
    parser.add_argument("--portfolio-id", action="append", default=[])
    parser.add_argument("--price", action="append", default=[])
    parser.add_argument("--fill-price", action="append", default=[])
    parser.add_argument("--record-fills", action="store_true")
    parser.add_argument("--generated-at")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_ml_config(args.config)
    ml_db = args.ml_db or str(config.data["ml_db"])
    generated_at = args.generated_at or datetime.now(UTC).isoformat()
    execution_date = args.execution_date or args.trade_date
    con = init_ml_db(ml_db)
    try:
        targets = _load_portfolio_targets(con, args.trade_date, args.portfolio_id)
        allocation = _load_allocation(con, args.trade_date)
        live_targets = build_live_target_positions(
            targets,
            allocation,
            trade_date=args.trade_date,
            account_id=args.account_id,
            account_nav=args.account_nav,
            generated_at=generated_at,
        )
        risk_logs = build_risk_logs(
            live_targets,
            trade_date=args.trade_date,
            account_id=args.account_id,
            strategy_id="phase10_live_pipeline",
            generated_at=generated_at,
        )
        prices = _parse_price_map(args.price)
        orders = build_live_orders(
            live_targets,
            execution_date=execution_date,
            prices=prices,
            generated_at=generated_at,
        )
        fills = record_live_fills(
            orders,
            fill_prices=_parse_price_map(args.fill_price) or prices,
            fill_time=generated_at,
        ) if args.record_fills else pd.DataFrame()

        upsert_dataframe(con, "live_target_positions", live_targets, ["trade_date", "account_id", "strategy_id", "code"])
        upsert_dataframe(con, "live_risk_logs", risk_logs, ["trade_date", "account_id", "strategy_id", "check_name"])
        upsert_dataframe(con, "live_orders", orders, ["order_id"])
        upsert_dataframe(con, "live_fills", fills, ["fill_id"])
    finally:
        con.close()
    print(f"live_targets={len(live_targets)} live_orders={len(orders)} live_fills={len(fills)} risk_logs={len(risk_logs)}")


def _load_portfolio_targets(con, trade_date: str, portfolio_ids: list[str]) -> pd.DataFrame:
    params: list[object] = [trade_date]
    where = "trade_date = ?"
    if portfolio_ids:
        placeholders = ", ".join("?" for _ in portfolio_ids)
        where += f" and portfolio_id in ({placeholders})"
        params.extend(portfolio_ids)
    return con.execute(
        f"""
        select *
        from ml_portfolio_targets_daily
        where {where}
        order by portfolio_id, code
        """,
        params,
    ).fetchdf()


def _load_allocation(con, trade_date: str) -> pd.DataFrame:
    return con.execute(
        """
        select *
        from ml_strategy_allocation_daily
        where trade_date = ?
        """,
        [trade_date],
    ).fetchdf()


def _parse_price_map(items: list[str]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"invalid price mapping: {item}")
        code, value = item.split("=", 1)
        prices[code] = float(value)
    return prices


if __name__ == "__main__":
    main()
