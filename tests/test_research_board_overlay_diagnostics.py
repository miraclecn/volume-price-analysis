from __future__ import annotations

import pandas as pd
import pytest

from scripts.research_board_overlay_diagnostics import overlay_diagnostics, worst_days_table, yearly_contribution


def test_overlay_diagnostics_reports_correlation_and_down_day_contribution() -> None:
    nav = _nav()

    diag = overlay_diagnostics(nav).iloc[0]

    assert diag["rows"] == 4
    assert diag["board_mean_return_on_main_down_days"] == pytest.approx(-0.001)
    assert diag["both_down_rate"] == pytest.approx(0.25)
    assert diag["worst_combined_day_return"] == pytest.approx(-0.018)


def test_yearly_contribution_groups_by_year() -> None:
    nav = _nav()

    yearly = yearly_contribution(nav)

    assert yearly["period"].tolist() == [2024, 2025]
    assert yearly.iloc[0]["board_return_sum"] == pytest.approx(0.004)


def test_worst_days_table_orders_by_combined_return() -> None:
    nav = _nav()

    worst = worst_days_table(nav, n=2)

    assert worst["trade_date"].tolist() == ["2024-01-03", "2025-01-02"]


def _nav() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "main_return": 0.01, "board_return": 0.002, "combined_return": 0.012, "drawdown": 0.0},
            {"trade_date": "2024-01-03", "main_return": -0.02, "board_return": 0.002, "combined_return": -0.018, "drawdown": -0.018},
            {"trade_date": "2025-01-02", "main_return": -0.01, "board_return": -0.004, "combined_return": -0.014, "drawdown": -0.014},
            {"trade_date": "2025-01-03", "main_return": 0.02, "board_return": 0.006, "combined_return": 0.026, "drawdown": 0.0},
        ]
    )
