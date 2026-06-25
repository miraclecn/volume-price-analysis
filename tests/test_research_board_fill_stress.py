from __future__ import annotations

import pandas as pd
import pytest

from scripts.research_board_fill_stress import FillStressVariant, run_fill_stress_backtest


def test_fill_stress_adverse_mode_fills_worst_realized_returns_inside_top_candidates() -> None:
    predictions = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "best_pred_bad", "pred_ret": 0.05, "pred_win_prob": 0.9, "target_ret_net": -0.10},
            {"trade_date": "2024-01-02", "code": "second_good", "pred_ret": 0.04, "pred_win_prob": 0.8, "target_ret_net": 0.10},
            {"trade_date": "2024-01-02", "code": "third_mid", "pred_ret": 0.03, "pred_win_prob": 0.7, "target_ret_net": 0.02},
        ]
    )
    variant = FillStressVariant(
        name="adverse",
        max_candidates=3,
        max_fills=1,
        name_weight=0.05,
        total_exposure_cap=0.05,
        fill_mode="adverse_realized",
    )

    result = run_fill_stress_backtest(predictions, variant, initial_nav=100_000.0)
    orders = result["orders"]

    assert orders["code"].tolist() == ["best_pred_bad"]
    assert orders.iloc[0]["pnl"] == pytest.approx(-500.0)
    assert result["nav"].iloc[-1]["nav"] == pytest.approx(99_500.0)


def test_fill_stress_extra_slippage_reduces_adjusted_return() -> None:
    predictions = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "pred_ret": 0.05, "pred_win_prob": 0.9, "target_ret_net": 0.02},
        ]
    )
    variant = FillStressVariant(
        name="slip",
        max_candidates=1,
        max_fills=1,
        name_weight=0.05,
        total_exposure_cap=0.05,
        fill_mode="top_pred",
        extra_slippage_bps=100.0,
    )

    result = run_fill_stress_backtest(predictions, variant, initial_nav=100_000.0)

    assert result["orders"].iloc[0]["adjusted_ret_net"] == pytest.approx(0.01)
