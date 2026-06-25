from __future__ import annotations

import pandas as pd
import pytest

from scripts.research_board_fill_aware_selection import (
    FillAwareVariant,
    run_fill_aware_backtest,
    select_candidates,
)
from scripts.research_board_fill_proxy_sweep import prepare_fill_proxy_frame


def test_select_candidates_expected_pred_ret_balances_alpha_and_fill_probability() -> None:
    predictions = prepare_fill_proxy_frame(_predictions())
    variant = FillAwareVariant(
        name="expected",
        fill_probs={"turn_q1": 0.10, "turn_q2": 0.20, "turn_q3": 0.40, "turn_q4": 0.70, "turn_q5": 0.90},
        selection_policy="expected_pred_ret",
        max_candidates=2,
    )

    selected = select_candidates(predictions, variant)

    assert selected["code"].tolist() == ["easier", "mid"]


def test_select_candidates_low_turnover_policy_prefers_low_turnover_even_with_lower_alpha() -> None:
    predictions = prepare_fill_proxy_frame(_predictions())
    variant = FillAwareVariant(
        name="low_turnover",
        fill_probs={f"turn_q{i}": 1.0 for i in range(1, 6)},
        selection_policy="low_turnover_first",
        max_candidates=2,
    )

    selected = select_candidates(predictions, variant)

    assert selected["code"].tolist() == ["hard_best", "hard_second"]


def test_fill_aware_backtest_uses_selected_policy() -> None:
    predictions = prepare_fill_proxy_frame(_predictions())
    alpha = FillAwareVariant(
        name="alpha",
        fill_probs={"turn_q1": 0.10, "turn_q2": 0.20, "turn_q3": 0.40, "turn_q4": 0.70, "turn_q5": 0.90},
        selection_policy="alpha_top",
        max_candidates=1,
        name_weight=0.10,
        total_exposure_cap=0.10,
    )
    expected = FillAwareVariant(
        name="expected",
        fill_probs={"turn_q1": 0.10, "turn_q2": 0.20, "turn_q3": 0.40, "turn_q4": 0.70, "turn_q5": 0.90},
        selection_policy="expected_pred_ret",
        max_candidates=1,
        name_weight=0.10,
        total_exposure_cap=0.10,
    )

    alpha_result = run_fill_aware_backtest(predictions, alpha, initial_nav=100_000.0)
    expected_result = run_fill_aware_backtest(predictions, expected, initial_nav=100_000.0)

    assert alpha_result["orders"]["code"].tolist() == ["hard_best"]
    assert expected_result["orders"]["code"].tolist() == ["easier"]
    assert expected_result["nav"].iloc[-1]["nav"] > alpha_result["nav"].iloc[-1]["nav"]
    assert expected_result["daily"].iloc[0]["expected_fill_count"] == pytest.approx(0.7)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "hard_best", "target_ret_net": 0.10, "pred_ret": 0.10, "pred_win_prob": 0.9, "turnover_rate": 1.0},
            {"trade_date": "2024-01-02", "code": "hard_second", "target_ret_net": 0.08, "pred_ret": 0.08, "pred_win_prob": 0.8, "turnover_rate": 2.0},
            {"trade_date": "2024-01-02", "code": "mid", "target_ret_net": 0.06, "pred_ret": 0.06, "pred_win_prob": 0.7, "turnover_rate": 3.0},
            {"trade_date": "2024-01-02", "code": "easier", "target_ret_net": 0.04, "pred_ret": 0.04, "pred_win_prob": 0.6, "turnover_rate": 4.0},
            {"trade_date": "2024-01-02", "code": "easy_weak", "target_ret_net": 0.01, "pred_ret": 0.01, "pred_win_prob": 0.5, "turnover_rate": 5.0},
        ]
    )
