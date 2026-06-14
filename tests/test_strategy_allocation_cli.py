from __future__ import annotations

import pandas as pd

from scripts.run_strategy_allocation import build_arg_parser, main
from ml_stock_selector.storage import init_ml_db, upsert_dataframe


def test_strategy_allocation_cli_accepts_phase9_inputs():
    args = build_arg_parser().parse_args(
        [
            "--trade-date",
            "2026-06-12",
            "--final-regime",
            "risk_on",
            "--account-drawdown",
            "-0.12",
            "--core-bundle-id",
            "core_bundle",
        ]
    )

    assert args.trade_date == "2026-06-12"
    assert args.final_regime == "risk_on"
    assert args.account_drawdown == -0.12
    assert args.core_bundle_id == "core_bundle"


def test_strategy_allocation_cli_writes_allocation_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "ml.duckdb"
    con = init_ml_db(db_path)
    upsert_dataframe(
        con,
        "ml_model_health_daily",
        pd.DataFrame(
            [
                {
                    "trade_date": "2026-06-12",
                    "model_or_bundle_id": "core_bundle",
                    "strategy_id": "holding_aware_v2",
                    "score_version": "v2_absolute_risk_filter",
                    "enabled_by_health": True,
                    "reason": "healthy",
                },
                {
                    "trade_date": "2026-06-12",
                    "model_or_bundle_id": "aggressive_bundle",
                    "strategy_id": "holding_aware_v2",
                    "score_version": "v2_three_model",
                    "enabled_by_health": False,
                    "reason": "rolling_drawdown_breached",
                },
            ]
        ),
        ["trade_date", "model_or_bundle_id", "strategy_id", "score_version"],
    )
    con.close()
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_strategy_allocation.py",
            "--ml-db",
            str(db_path),
            "--trade-date",
            "2026-06-12",
            "--final-regime",
            "risk_on",
            "--account-drawdown",
            "-0.12",
            "--core-bundle-id",
            "core_bundle",
            "--aggressive-bundle-id",
            "aggressive_bundle",
        ],
    )

    main()

    check = init_ml_db(db_path)
    rows = check.execute(
        """
        select sleeve, final_weight
        from ml_strategy_allocation_daily
        where trade_date = '2026-06-12'
        order by sleeve
        """
    ).fetchall()
    check.close()
    assert rows == [
        ("aggressive", 0.0),
        ("cash", 0.675),
        ("core", 0.275),
        ("fixed_horizon", 0.05),
    ]
