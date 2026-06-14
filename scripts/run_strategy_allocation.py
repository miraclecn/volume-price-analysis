from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.storage import init_ml_db, upsert_dataframe
from ml_stock_selector.strategy.allocation import allocate_strategy_sleeves
from ml_stock_selector.strategy.ensemble import default_phase9_sleeves


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--ml-db")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--final-regime", choices=["risk_on", "neutral", "risk_off", "crash"], required=True)
    parser.add_argument("--account-drawdown", type=float, default=0.0)
    parser.add_argument("--core-bundle-id")
    parser.add_argument("--aggressive-bundle-id")
    parser.add_argument("--fixed-horizon-bundle-id")
    parser.add_argument("--generated-at")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_ml_config(args.config)
    ml_db = args.ml_db or str(config.data["ml_db"])
    con = init_ml_db(ml_db)
    try:
        sleeves = default_phase9_sleeves(
            core_bundle_id=args.core_bundle_id,
            aggressive_bundle_id=args.aggressive_bundle_id,
            fixed_horizon_bundle_id=args.fixed_horizon_bundle_id,
        )
        health = _load_health_by_bundle(
            con,
            args.trade_date,
            [sleeve.bundle_id for sleeve in sleeves if sleeve.bundle_id],
        )
        rows = allocate_strategy_sleeves(
            trade_date=args.trade_date,
            sleeves=sleeves,
            final_regime=args.final_regime,
            account_drawdown=args.account_drawdown,
            health_enabled_by_bundle=health,
            generated_at=args.generated_at or datetime.now(UTC).isoformat(),
        )
        upsert_dataframe(
            con,
            "ml_strategy_allocation_daily",
            rows,
            ["trade_date", "strategy_id", "sleeve", "bundle_id", "score_version"],
        )
    finally:
        con.close()
    print(f"allocation_rows={len(rows)}")


def _load_health_by_bundle(con, trade_date: str, bundle_ids: list[str]) -> dict[str, bool]:
    if not bundle_ids:
        return {}
    placeholders = ", ".join("?" for _ in bundle_ids)
    frame = con.execute(
        f"""
        select model_or_bundle_id, enabled_by_health
        from ml_model_health_daily
        where trade_date = ?
          and model_or_bundle_id in ({placeholders})
        """,
        [trade_date, *bundle_ids],
    ).fetchdf()
    if frame.empty:
        return {}
    frame["enabled_by_health"] = frame["enabled_by_health"].fillna(True)
    return {
        str(row.model_or_bundle_id): bool(row.enabled_by_health)
        for row in frame.itertuples(index=False)
    }


if __name__ == "__main__":
    main()
