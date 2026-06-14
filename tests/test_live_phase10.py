from __future__ import annotations

import pandas as pd

from ml_stock_selector.serving.live import (
    build_live_orders,
    build_live_target_positions,
    build_risk_logs,
    record_live_fills,
)


def test_build_live_target_positions_preserves_sleeve_lineage():
    targets = pd.DataFrame(
        [
            {
                "trade_date": "2026-06-12",
                "portfolio_id": "core_portfolio",
                "score_version": "v2_absolute_risk_filter",
                "code": "000001.SZ",
                "target_weight": 0.10,
                "entry_reason": "core_pool",
            },
            {
                "trade_date": "2026-06-12",
                "portfolio_id": "aggressive_portfolio",
                "score_version": "v2_three_model",
                "code": "000002.SZ",
                "target_weight": 0.20,
                "entry_reason": "candidate_pool",
            },
        ]
    )
    allocation = pd.DataFrame(
        [
            {
                "trade_date": "2026-06-12",
                "strategy_id": "core_portfolio",
                "sleeve": "core",
                "bundle_id": "core_bundle",
                "score_version": "v2_absolute_risk_filter",
                "final_weight": 0.50,
            },
            {
                "trade_date": "2026-06-12",
                "strategy_id": "aggressive_portfolio",
                "sleeve": "aggressive",
                "bundle_id": "aggressive_bundle",
                "score_version": "v2_three_model",
                "final_weight": 0.0,
            },
        ]
    )

    rows = build_live_target_positions(
        targets,
        allocation,
        trade_date="2026-06-12",
        account_id="paper",
        account_nav=1_000_000.0,
        generated_at="t",
    )

    assert rows[["code", "source_sleeve", "source_bundle_id", "target_weight", "target_value"]].to_dict("records") == [
        {
            "code": "000001.SZ",
            "source_sleeve": "core",
            "source_bundle_id": "core_bundle",
            "target_weight": 0.05,
            "target_value": 50000.0,
        },
        {
            "code": "000002.SZ",
            "source_sleeve": "aggressive",
            "source_bundle_id": "aggressive_bundle",
            "target_weight": 0.0,
            "target_value": 0.0,
        },
    ]


def test_live_orders_and_fills_are_traceable_to_targets():
    targets = pd.DataFrame(
        [
            {
                "trade_date": "2026-06-12",
                "account_id": "paper",
                "strategy_id": "core_portfolio",
                "code": "000001.SZ",
                "target_weight": 0.05,
                "target_value": 50000.0,
                "source_bundle_id": "core_bundle",
                "source_sleeve": "core",
                "score_version": "v2_absolute_risk_filter",
                "reason": "core_pool",
            }
        ]
    )

    orders = build_live_orders(
        targets,
        execution_date="2026-06-13",
        prices={"000001.SZ": 10.0},
        generated_at="t",
    )
    fills = record_live_fills(
        orders,
        fill_prices={"000001.SZ": 10.05},
        fill_time="2026-06-13T09:31:00",
    )

    assert orders.iloc[0]["order_id"] == "paper_20260613_core_portfolio_000001_SZ_buy"
    assert orders.iloc[0]["order_qty"] == 5000.0
    assert fills.iloc[0]["order_id"] == orders.iloc[0]["order_id"]
    assert fills.iloc[0]["slippage_bps"] == 50.0


def test_build_risk_logs_records_no_signal_status():
    logs = build_risk_logs(
        pd.DataFrame(),
        trade_date="2026-06-12",
        account_id="paper",
        strategy_id="phase10_live",
        generated_at="t",
    )

    assert logs.iloc[0]["check_name"] == "target_positions_present"
    assert logs.iloc[0]["passed"] is False
    assert logs.iloc[0]["action"] == "no_order"
