from __future__ import annotations

import pandas as pd
import pytest

from scripts.research_board_market_position_sweep import MarketPositionVariant, run_market_position_backtest


def test_market_position_backtest_reduces_exposure_in_weak_prev_up_regime() -> None:
    predictions = pd.DataFrame(
        [
            _row("2024-01-02", "a", 0.10, prev_up_ratio=0.34),
            _row("2024-01-03", "b", 0.10, prev_up_ratio=0.50),
        ]
    )
    variant = MarketPositionVariant(
        name="prevup",
        max_positions=1,
        name_weight=0.10,
        total_exposure_cap=0.10,
        market_rule="prev_up",
        prev_up_zero_below=0.30,
        prev_up_half_below=0.40,
    )

    result = run_market_position_backtest(predictions, variant, initial_nav=100_000.0)

    assert result["daily"]["exposure"].tolist() == [0.5, 1.0]
    assert result["orders"]["weight"].tolist() == pytest.approx([0.05, 0.10])


def test_market_position_backtest_can_half_size_outside_heat_window() -> None:
    predictions = pd.DataFrame(
        [
            _row("2024-01-02", "cold", 0.10, prev_sealed_count=10),
            _row("2024-01-03", "hot", 0.10, prev_sealed_count=80),
        ]
    )
    variant = MarketPositionVariant(
        name="heat",
        max_positions=1,
        name_weight=0.10,
        total_exposure_cap=0.10,
        market_rule="heat",
        heat_min=20,
        heat_max=100,
        heat_half_outside=True,
    )

    result = run_market_position_backtest(predictions, variant, initial_nav=100_000.0)

    assert result["daily"]["exposure"].tolist() == [0.5, 1.0]
    assert result["orders"]["code"].tolist() == ["cold", "hot"]


def test_market_position_backtest_confidence_scale_uses_predicted_return() -> None:
    predictions = pd.DataFrame(
        [
            _row("2024-01-02", "low_conf", 0.10, pred_ret=0.010),
            _row("2024-01-03", "high_conf", -0.10, pred_ret=0.050),
        ]
    )
    variant = MarketPositionVariant(
        name="confidence",
        max_positions=1,
        name_weight=0.10,
        total_exposure_cap=0.10,
        confidence_scale=True,
    )

    result = run_market_position_backtest(predictions, variant, initial_nav=100_000.0)

    assert result["daily"]["exposure"].tolist() == [0.5, 1.5]
    assert result["orders"]["weight"].tolist() == pytest.approx([0.05, 0.15])


def test_market_position_backtest_scales_exposure_by_board_heat() -> None:
    predictions = pd.DataFrame(
        [
            _row("2024-01-02", "cold", 0.10, prev_sealed_count=20),
            _row("2024-01-03", "normal", 0.10, prev_sealed_count=60),
            _row("2024-01-04", "hot", 0.10, prev_sealed_count=120),
        ]
    )
    variant = MarketPositionVariant(
        name="heat_scale",
        max_positions=1,
        name_weight=0.10,
        total_exposure_cap=0.10,
        market_rule="heat_scale",
    )

    result = run_market_position_backtest(predictions, variant, initial_nav=100_000.0)

    assert result["daily"]["exposure"].tolist() == [0.5, 1.0, 1.5]
    assert result["orders"]["weight"].tolist() == pytest.approx([0.05, 0.10, 0.15])


def _row(
    trade_date: str,
    code: str,
    target_ret_net: float,
    *,
    pred_ret: float = 0.03,
    pred_win_prob: float = 0.70,
    prev_up_ratio: float = 0.50,
    prev_sealed_count: float = 60,
) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "code": code,
        "target_ret_net": target_ret_net,
        "target_win": int(target_ret_net > 0),
        "second_board_success": target_ret_net > 0.08,
        "pred_ret": pred_ret,
        "pred_win_prob": pred_win_prob,
        "prev_up_ratio": prev_up_ratio,
        "prev_sealed_count": prev_sealed_count,
    }
