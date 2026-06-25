from __future__ import annotations

import pandas as pd
import pytest

from scripts.research_board_overnight_model import OvernightVariant, prepare_overnight_dataset, run_overnight_backtest


def test_prepare_overnight_dataset_keeps_only_sealed_boards_and_applies_trade_costs() -> None:
    raw = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "sealed",
                "open": 10.0,
                "high": 11.0,
                "low": 10.0,
                "close": 11.0,
                "prev_close": 10.0,
                "limit_up": 11.0,
                "limit_band": "limit_10pct",
                "amount": 100_000_000.0,
                "adv20_amount": 100_000_000.0,
                "turnover_rate": 5.0,
                "next_open": 11.55,
                "next_high": 12.1,
                "next_close": 12.1,
                "next_limit_up": 12.1,
                "next2_open": 12.0,
            },
            {
                "trade_date": "2024-01-02",
                "code": "failed",
                "open": 20.0,
                "high": 22.0,
                "low": 19.0,
                "close": 20.5,
                "prev_close": 20.0,
                "limit_up": 22.0,
                "limit_band": "limit_10pct",
                "amount": 100_000_000.0,
                "adv20_amount": 100_000_000.0,
                "turnover_rate": 5.0,
                "next_open": 20.0,
                "next_high": 20.5,
                "next_close": 20.2,
                "next_limit_up": 22.5,
                "next2_open": 20.1,
            },
        ]
    )

    out = prepare_overnight_dataset(raw, slippage_bps=10.0, commission_bps=3.0, stamp_duty_bps=5.0)

    assert out["code"].tolist() == ["sealed"]
    assert out.iloc[0]["target_ret_net"] == pytest.approx(11.55 / 11.0 - 1.0 - 0.0018)
    assert out.iloc[0]["target_win"] == 1


def test_run_overnight_backtest_selects_top_predictions_with_weight_caps() -> None:
    predictions = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "target_ret_net": 0.10, "target_win": 1, "second_board_success": True, "pred_ret": 0.03, "pred_win_prob": 0.7, "prev_up_ratio": 0.5, "prev_sealed_count": 50},
            {"trade_date": "2024-01-02", "code": "b", "target_ret_net": -0.10, "target_win": 0, "second_board_success": False, "pred_ret": 0.02, "pred_win_prob": 0.6, "prev_up_ratio": 0.5, "prev_sealed_count": 50},
            {"trade_date": "2024-01-02", "code": "c", "target_ret_net": 0.50, "target_win": 1, "second_board_success": True, "pred_ret": -0.01, "pred_win_prob": 0.2, "prev_up_ratio": 0.5, "prev_sealed_count": 50},
        ]
    )
    variant = OvernightVariant("test", max_positions=2, max_name_weight=0.2, max_total_exposure=0.3)

    result = run_overnight_backtest(predictions, variant, initial_nav=100_000.0)
    orders = result["orders"]

    assert orders["code"].tolist() == ["a", "b"]
    assert orders["weight"].tolist() == pytest.approx([0.15, 0.15])
    assert result["metrics"]["total_return"] == pytest.approx(0.0)
