from __future__ import annotations

import pandas as pd
import pytest

from scripts.research_board_taxonomy import add_board_taxonomy_labels, daily_market_board_stats


def test_board_taxonomy_distinguishes_sealed_failed_and_second_board() -> None:
    frame = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "sealed",
                "open": 10.0,
                "high": 11.0,
                "low": 9.8,
                "close": 11.0,
                "prev_close": 10.0,
                "limit_up": 11.0,
                "limit_band": "limit_10pct",
                "turnover_rate": 5.0,
                "next_open": 11.5,
                "next_high": 12.1,
                "next_close": 12.1,
                "next_limit_up": 12.1,
                "next2_open": 12.3,
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
                "turnover_rate": 10.0,
                "next_open": 20.0,
                "next_high": 20.8,
                "next_close": 20.2,
                "next_limit_up": 22.5,
                "next2_open": 19.8,
            },
        ]
    )

    out = add_board_taxonomy_labels(frame, slippage_bps=0.0, commission_bps=0.0, stamp_duty_bps=0.0).set_index("code")

    assert bool(out.loc["sealed", "touch_board_today"]) is True
    assert bool(out.loc["sealed", "sealed_today"]) is True
    assert bool(out.loc["sealed", "failed_board_today"]) is False
    assert bool(out.loc["sealed", "second_board_success"]) is True
    assert out.loc["sealed", "board_next_open_ret"] == pytest.approx(11.5 / 11.0 - 1.0)
    assert out.loc["sealed", "relay_next2_open_ret"] == pytest.approx(12.3 / 11.5 - 1.0)
    assert bool(out.loc["failed", "touch_board_today"]) is True
    assert bool(out.loc["failed", "sealed_today"]) is False
    assert bool(out.loc["failed", "failed_board_today"]) is True


def test_daily_market_board_stats_adds_previous_market_state() -> None:
    frame = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "open": 10.0, "close": 11.0, "prev_close": 10.0, "high": 11.0, "limit_up": 11.0, "next_high": None, "next_close": None, "next_limit_up": None, "next_open": None, "next2_open": None},
            {"trade_date": "2024-01-02", "code": "b", "open": 10.0, "close": 9.5, "prev_close": 10.0, "high": 10.0, "limit_up": 11.0, "next_high": None, "next_close": None, "next_limit_up": None, "next_open": None, "next2_open": None},
            {"trade_date": "2024-01-03", "code": "a", "open": 11.0, "close": 12.1, "prev_close": 11.0, "high": 12.1, "limit_up": 12.1, "next_high": None, "next_close": None, "next_limit_up": None, "next_open": None, "next2_open": None},
        ]
    )
    labeled = add_board_taxonomy_labels(frame, slippage_bps=0.0, commission_bps=0.0, stamp_duty_bps=0.0)

    daily = daily_market_board_stats(labeled).set_index("trade_date")

    assert daily.loc["2024-01-02", "up_ratio"] == pytest.approx(0.5)
    assert daily.loc["2024-01-02", "sealed_count"] == 1
    assert daily.loc["2024-01-03", "prev_up_ratio"] == pytest.approx(0.5)
    assert daily.loc["2024-01-03", "prev_sealed_count"] == 1
