from __future__ import annotations

import pandas as pd
import pytest

from scripts.research_board_signal_diagnostics import (
    SignalDiagnosticsConfig,
    pairwise_segments,
    prepare_signal_frame,
    single_factor_segments,
    top_candidate_profile,
)


def test_prepare_signal_frame_marks_top_candidates_by_daily_prediction_rank() -> None:
    frame = prepare_signal_frame(_predictions(), top_n=2)

    top_codes = frame[frame["is_top_candidate"]]["code"].tolist()

    assert top_codes == ["a", "b", "d", "e"]
    assert frame.loc[frame["code"].eq("a"), "daily_rank"].iloc[0] == 1


def test_signal_diagnostics_outputs_single_and_pairwise_segments() -> None:
    frame = prepare_signal_frame(_predictions(), top_n=2)
    config = SignalDiagnosticsConfig(top_n=2, min_segment_rows=1)

    single = single_factor_segments(frame, config=config)
    pairwise = pairwise_segments(frame, config=config)
    profile = top_candidate_profile(frame)

    assert {"all_sealed", "top2"}.issubset(set(single["universe"]))
    assert "heat_bucket+turnover_bucket" in set(pairwise["factor"])
    assert profile.set_index("universe").loc["top_candidate", "rows"] == 4
    assert profile.set_index("universe").loc["top_candidate", "target_ret_net_mean"] == pytest.approx(0.0425)


def test_prepare_signal_frame_requires_explanatory_columns() -> None:
    with pytest.raises(ValueError, match="missing prediction columns"):
        prepare_signal_frame(pd.DataFrame([{"trade_date": "2024-01-02"}]), top_n=2)


def _predictions() -> pd.DataFrame:
    rows = []
    for date, codes in [
        ("2024-01-02", [("a", 0.05, 0.08), ("b", 0.04, 0.04), ("c", 0.01, -0.02)]),
        ("2024-01-03", [("d", 0.06, 0.03), ("e", 0.03, 0.02), ("f", 0.02, -0.01)]),
    ]:
        for code, pred_ret, realized in codes:
            rows.append(
                {
                    "trade_date": date,
                    "code": code,
                    "target_ret_net": realized,
                    "target_win": int(realized > 0),
                    "second_board_success": realized > 0.05,
                    "ret_1": 0.10,
                    "turnover_rate": 5.0,
                    "adv20_amount": 100_000_000.0,
                    "limit_band_clean": "limit_10pct",
                    "prev_up_ratio": 0.50,
                    "prev_sealed_count": 60,
                    "pred_ret": pred_ret,
                    "pred_win_prob": 0.70,
                }
            )
    return pd.DataFrame(rows)
