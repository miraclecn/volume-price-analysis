from __future__ import annotations

import pandas as pd
import pytest

from scripts.research_board_fill_proxy_sweep import (
    FillProxyVariant,
    prepare_fill_proxy_frame,
    run_fill_proxy_backtest,
)


def test_fill_proxy_backtest_uses_turnover_bucket_probabilities_without_reallocating_unfilled_cash() -> None:
    predictions = prepare_fill_proxy_frame(_predictions())
    variant = FillProxyVariant(
        name="proxy",
        fill_probs={"turn_q1": 0.10, "turn_q2": 0.20, "turn_q3": 0.30, "turn_q4": 0.40, "turn_q5": 0.50},
        max_candidates=5,
        name_weight=0.01,
        total_exposure_cap=0.05,
    )

    result = run_fill_proxy_backtest(predictions, variant, initial_nav=100_000.0)

    assert result["daily"].iloc[0]["expected_fill_count"] == pytest.approx(1.5)
    assert result["daily"].iloc[0]["expected_exposure"] == pytest.approx(0.015)
    assert result["nav"].iloc[-1]["nav"] == pytest.approx(100_150.0)


def test_fill_proxy_extra_slippage_reduces_adjusted_returns() -> None:
    predictions = prepare_fill_proxy_frame(_predictions())
    no_slip = FillProxyVariant(name="no", fill_probs={f"turn_q{i}": 1.0 for i in range(1, 6)})
    slip = FillProxyVariant(name="slip", fill_probs={f"turn_q{i}": 1.0 for i in range(1, 6)}, extra_slippage_bps=100.0)

    no_slip_result = run_fill_proxy_backtest(predictions, no_slip, initial_nav=100_000.0)
    slip_result = run_fill_proxy_backtest(predictions, slip, initial_nav=100_000.0)

    assert slip_result["nav"].iloc[-1]["nav"] < no_slip_result["nav"].iloc[-1]["nav"]
    assert slip_result["orders"].iloc[0]["adjusted_ret_net"] == pytest.approx(0.09)


def test_fill_proxy_min_pred_ret_filters_attempts() -> None:
    predictions = prepare_fill_proxy_frame(_predictions())
    variant = FillProxyVariant(
        name="filtered",
        fill_probs={f"turn_q{i}": 1.0 for i in range(1, 6)},
        max_candidates=5,
        min_pred_ret=0.04,
    )

    result = run_fill_proxy_backtest(predictions, variant, initial_nav=100_000.0)

    assert result["daily"].iloc[0]["attempt_count"] == 2
    assert result["orders"]["code"].tolist() == ["a", "b"]


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "target_ret_net": 0.10, "pred_ret": 0.05, "pred_win_prob": 0.9, "turnover_rate": 1.0},
            {"trade_date": "2024-01-02", "code": "b", "target_ret_net": 0.10, "pred_ret": 0.04, "pred_win_prob": 0.8, "turnover_rate": 2.0},
            {"trade_date": "2024-01-02", "code": "c", "target_ret_net": 0.10, "pred_ret": 0.03, "pred_win_prob": 0.7, "turnover_rate": 3.0},
            {"trade_date": "2024-01-02", "code": "d", "target_ret_net": 0.10, "pred_ret": 0.02, "pred_win_prob": 0.6, "turnover_rate": 4.0},
            {"trade_date": "2024-01-02", "code": "e", "target_ret_net": 0.10, "pred_ret": 0.01, "pred_win_prob": 0.5, "turnover_rate": 5.0},
        ]
    )
