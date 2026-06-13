from __future__ import annotations

import pandas as pd

from ml_stock_selector.backtest.reports import fixed_horizon_exit_reason_report


def test_fixed_horizon_exit_reason_report_summarizes_realized_returns():
    orders = pd.DataFrame(
        {
            "side": ["sell", "sell", "buy"],
            "status": ["filled", "filled", "filled"],
            "exit_reason": ["time_exit", "risk_exit", None],
            "realized_ret": [0.05, -0.02, None],
        }
    )

    report = fixed_horizon_exit_reason_report(orders)

    by_reason = report.set_index("exit_reason")
    assert by_reason.loc["time_exit", "sell_count"] == 1
    assert by_reason.loc["risk_exit", "avg_realized_ret"] == -0.02
