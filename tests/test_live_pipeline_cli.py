from __future__ import annotations

import pandas as pd

from ml_stock_selector.storage import init_ml_db, upsert_dataframe
from scripts.run_live_pipeline import build_arg_parser, main


def test_live_pipeline_cli_accepts_phase10_arguments():
    args = build_arg_parser().parse_args(
        [
            "--trade-date",
            "2026-06-12",
            "--account-id",
            "paper",
            "--account-nav",
            "1000000",
            "--portfolio-id",
            "core_portfolio",
        ]
    )

    assert args.trade_date == "2026-06-12"
    assert args.account_id == "paper"
    assert args.account_nav == 1_000_000.0
    assert args.portfolio_id == ["core_portfolio"]


def test_live_pipeline_cli_writes_targets_risk_logs_and_orders(tmp_path, monkeypatch):
    db_path = tmp_path / "ml.duckdb"
    con = init_ml_db(db_path)
    upsert_dataframe(
        con,
        "ml_portfolio_targets_daily",
        pd.DataFrame(
            [
                {
                    "trade_date": "2026-06-12",
                    "run_id": "daily",
                    "fold_id": "daily",
                    "portfolio_id": "core_portfolio",
                    "score_version": "v2_absolute_risk_filter",
                    "code": "000001.SZ",
                    "target_weight": 0.10,
                    "entry_reason": "core_pool",
                }
            ]
        ),
        ["trade_date", "run_id", "fold_id", "portfolio_id", "score_version", "code"],
    )
    upsert_dataframe(
        con,
        "ml_strategy_allocation_daily",
        pd.DataFrame(
            [
                {
                    "trade_date": "2026-06-12",
                    "strategy_id": "core_portfolio",
                    "sleeve": "core",
                    "bundle_id": "core_bundle",
                    "score_version": "v2_absolute_risk_filter",
                    "final_weight": 0.50,
                }
            ]
        ),
        ["trade_date", "strategy_id", "sleeve", "bundle_id", "score_version"],
    )
    con.close()
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_live_pipeline.py",
            "--ml-db",
            str(db_path),
            "--trade-date",
            "2026-06-12",
            "--execution-date",
            "2026-06-13",
            "--account-id",
            "paper",
            "--account-nav",
            "1000000",
            "--portfolio-id",
            "core_portfolio",
            "--price",
            "000001.SZ=10",
            "--generated-at",
            "t",
        ],
    )

    main()

    check = init_ml_db(db_path)
    target = check.execute(
        "select source_sleeve, source_bundle_id, target_weight, target_value from live_target_positions"
    ).fetchone()
    order = check.execute("select code, order_qty, status from live_orders").fetchone()
    risk = check.execute("select check_name, passed, action from live_risk_logs").fetchone()
    check.close()
    assert target == ("core", "core_bundle", 0.05, 50000.0)
    assert order == ("000001.SZ", 5000.0, "created")
    assert risk == ("target_positions_present", True, "allow")
