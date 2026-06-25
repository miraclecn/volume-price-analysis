from __future__ import annotations

import pandas as pd
import pytest

from scripts.research_limit_hit_strategy import build_limit_hit_labels, run_limit_hit_backtest


def test_limit_hit_label_uses_next_trading_day_high_against_next_limit_up() -> None:
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "000001.SZ", "open": 10.0, "high": 10.2, "close": 10.1, "limit_up": 11.0, "is_paused": False},
        {"trade_date": "2024-01-03", "code": "000001.SZ", "open": 10.3, "high": 11.1, "close": 11.1, "limit_up": 11.1, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "000001.SZ", "open": 11.2, "high": 11.3, "close": 11.0, "limit_up": 12.2, "is_paused": False},
            {"trade_date": "2024-01-02", "code": "000002.SZ", "open": 20.0, "high": 20.3, "close": 20.1, "limit_up": 22.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "000002.SZ", "open": 20.2, "high": 21.0, "close": 20.8, "limit_up": 22.2, "is_paused": False},
        ]
    )

    labels = build_limit_hit_labels(bars).set_index(["trade_date", "code"])

    assert labels.loc[("2024-01-02", "000001.SZ"), "hit_limit_next_day"] == 1
    assert labels.loc[("2024-01-02", "000002.SZ"), "hit_limit_next_day"] == 0
    assert labels.loc[("2024-01-02", "000001.SZ"), "next_trade_date"] == "2024-01-03"
    assert labels.loc[("2024-01-02", "000001.SZ"), "next_open_ret"] == pytest.approx(0.2 / 10.1)


def test_limit_hit_label_can_require_next_day_close_at_limit_up() -> None:
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "000001.SZ", "open": 10.0, "high": 10.2, "close": 10.1, "limit_up": 11.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "000001.SZ", "open": 10.2, "high": 11.1, "close": 10.8, "limit_up": 11.1, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "000001.SZ", "open": 10.7, "high": 10.8, "close": 10.6, "limit_up": 11.9, "is_paused": False},
        ]
    )

    touch = build_limit_hit_labels(bars, limit_success_mode="touch").set_index(["trade_date", "code"])
    close = build_limit_hit_labels(bars, limit_success_mode="close").set_index(["trade_date", "code"])

    assert touch.loc[("2024-01-02", "000001.SZ"), "hit_limit_next_day"] == 1
    assert close.loc[("2024-01-02", "000001.SZ"), "hit_limit_next_day"] == 0


def test_limit_hit_backtest_sells_one_day_later_after_a_limit_hit() -> None:
    predictions = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "hit", "p_limit_hit": 0.9, "risk_prob": 0.1, "adv20_amount": 30_000_000.0},
            {"trade_date": "2024-01-02", "code": "miss", "p_limit_hit": 0.8, "risk_prob": 0.1, "adv20_amount": 30_000_000.0},
        ]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "hit", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "hit", "open": 10.0, "high": 11.0, "low": 9.9, "close": 11.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "hit", "open": 12.0, "high": 12.1, "low": 11.8, "close": 12.0, "limit_up": 12.1, "limit_down": 9.9, "is_paused": False},
            {"trade_date": "2024-01-05", "code": "hit", "open": 12.3, "high": 12.4, "low": 12.1, "close": 12.2, "limit_up": 13.2, "limit_down": 10.8, "is_paused": False},
            {"trade_date": "2024-01-02", "code": "miss", "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0, "limit_up": 22.0, "limit_down": 18.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "miss", "open": 20.0, "high": 20.5, "low": 19.8, "close": 20.1, "limit_up": 22.0, "limit_down": 18.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "miss", "open": 19.0, "high": 19.2, "low": 18.8, "close": 19.0, "limit_up": 22.1, "limit_down": 18.1, "is_paused": False},
        ]
    )

    result = run_limit_hit_backtest(
        predictions,
        bars,
        initial_cash=100_000.0,
        max_positions=2,
        min_probability=0.5,
        max_risk_prob=0.5,
        slippage_bps=0.0,
        commission_bps=0.0,
        stamp_duty_bps=0.0,
    )
    orders = result["orders"]

    hit_sell = orders[(orders["code"] == "hit") & (orders["side"] == "sell")].iloc[0]
    miss_sell = orders[(orders["code"] == "miss") & (orders["side"] == "sell")].iloc[0]
    assert hit_sell["trade_date"] == "2024-01-05"
    assert hit_sell["exit_reason"] == "limit_hit_delay_exit"
    assert miss_sell["trade_date"] == "2024-01-04"
    assert miss_sell["exit_reason"] == "miss_limit_exit"
    assert result["metrics"]["total_return"] > 0.0


def test_limit_hit_backtest_close_mode_does_not_delay_intraday_touch_only() -> None:
    predictions = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "touch", "p_limit_hit": 0.9, "risk_prob": 0.1, "adv20_amount": 30_000_000.0},
        ]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "touch", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "touch", "open": 10.0, "high": 11.0, "low": 9.9, "close": 10.5, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "touch", "open": 9.8, "high": 10.0, "low": 9.7, "close": 9.9, "limit_up": 11.5, "limit_down": 9.5, "is_paused": False},
            {"trade_date": "2024-01-05", "code": "touch", "open": 12.0, "high": 12.1, "low": 11.8, "close": 12.0, "limit_up": 13.2, "limit_down": 10.8, "is_paused": False},
        ]
    )

    result = run_limit_hit_backtest(
        predictions,
        bars,
        initial_cash=100_000.0,
        max_positions=1,
        min_probability=0.5,
        max_risk_prob=0.5,
        limit_success_mode="close",
        limit_hit_extra_hold_days=1,
        slippage_bps=0.0,
        commission_bps=0.0,
        stamp_duty_bps=0.0,
    )

    sell = result["orders"][result["orders"]["side"] == "sell"].iloc[0]
    assert sell["trade_date"] == "2024-01-04"
    assert sell["exit_reason"] == "miss_limit_exit"


def test_limit_hit_backtest_retries_sell_after_paused_exit_day() -> None:
    predictions = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "paused", "p_limit_hit": 0.9, "risk_prob": 0.1, "adv20_amount": 30_000_000.0},
        ]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "paused", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "paused", "open": 10.0, "high": 10.5, "low": 9.9, "close": 10.1, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "paused", "open": 9.8, "high": 9.8, "low": 9.8, "close": 9.8, "limit_up": 11.1, "limit_down": 9.1, "is_paused": True},
            {"trade_date": "2024-01-05", "code": "paused", "open": 9.7, "high": 9.9, "low": 9.6, "close": 9.8, "limit_up": 10.8, "limit_down": 8.8, "is_paused": False},
        ]
    )

    result = run_limit_hit_backtest(
        predictions,
        bars,
        initial_cash=100_000.0,
        max_positions=1,
        min_probability=0.5,
        max_risk_prob=0.5,
        slippage_bps=0.0,
        commission_bps=0.0,
        stamp_duty_bps=0.0,
    )

    sell = result["orders"][result["orders"]["side"] == "sell"].iloc[0]
    assert sell["trade_date"] == "2024-01-05"
    assert sell["exit_reason"] == "miss_limit_exit"


def test_limit_hit_backtest_can_filter_entry_day_return() -> None:
    predictions = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "weak", "p_limit_hit": 0.9, "risk_prob": 0.1, "ret_1": 0.05},
            {"trade_date": "2024-01-02", "code": "strong", "p_limit_hit": 0.8, "risk_prob": 0.1, "ret_1": 0.11},
        ]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "weak", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "weak", "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "weak", "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-02", "code": "strong", "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0, "limit_up": 22.0, "limit_down": 18.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "strong", "open": 20.0, "high": 20.1, "low": 19.9, "close": 20.0, "limit_up": 22.0, "limit_down": 18.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "strong", "open": 20.0, "high": 20.1, "low": 19.9, "close": 20.0, "limit_up": 22.0, "limit_down": 18.0, "is_paused": False},
        ]
    )

    result = run_limit_hit_backtest(
        predictions,
        bars,
        initial_cash=100_000.0,
        max_positions=1,
        min_probability=0.5,
        max_risk_prob=0.5,
        entry_min_ret=0.10,
        slippage_bps=0.0,
        commission_bps=0.0,
        stamp_duty_bps=0.0,
    )

    buys = result["orders"][result["orders"]["side"] == "buy"]
    assert buys["code"].tolist() == ["strong"]


def test_limit_hit_backtest_caps_single_position_weight() -> None:
    predictions = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "miss", "p_limit_hit": 0.9, "risk_prob": 0.1, "adv20_amount": 30_000_000.0},
        ]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "miss", "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0, "limit_up": 22.0, "limit_down": 18.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "miss", "open": 20.0, "high": 20.5, "low": 19.8, "close": 20.1, "limit_up": 22.0, "limit_down": 18.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "miss", "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "limit_up": 22.1, "limit_down": 9.0, "is_paused": False},
        ]
    )

    uncapped = run_limit_hit_backtest(
        predictions,
        bars,
        initial_cash=100_000.0,
        max_positions=1,
        min_probability=0.5,
        max_risk_prob=0.5,
        max_position_weight=1.0,
        slippage_bps=0.0,
        commission_bps=0.0,
        stamp_duty_bps=0.0,
    )
    capped = run_limit_hit_backtest(
        predictions,
        bars,
        initial_cash=100_000.0,
        max_positions=1,
        min_probability=0.5,
        max_risk_prob=0.5,
        max_position_weight=0.2,
        slippage_bps=0.0,
        commission_bps=0.0,
        stamp_duty_bps=0.0,
    )

    assert uncapped["metrics"]["max_drawdown"] < -0.45
    assert capped["metrics"]["max_drawdown"] > -0.12


def test_limit_hit_backtest_applies_market_exposure_scalar_to_entry_size() -> None:
    predictions = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "board", "p_limit_hit": 0.9, "risk_prob": 0.1, "adv20_amount": 30_000_000.0},
        ]
    )
    bars = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "board", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-03", "code": "board", "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
            {"trade_date": "2024-01-04", "code": "board", "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "limit_up": 11.0, "limit_down": 9.0, "is_paused": False},
        ]
    )

    full = run_limit_hit_backtest(
        predictions,
        bars,
        initial_cash=100_000.0,
        max_positions=1,
        max_position_weight=0.6,
        min_probability=0.5,
        max_risk_prob=0.5,
        slippage_bps=0.0,
        commission_bps=0.0,
        stamp_duty_bps=0.0,
    )
    half = run_limit_hit_backtest(
        predictions,
        bars,
        initial_cash=100_000.0,
        max_positions=1,
        max_position_weight=0.6,
        min_probability=0.5,
        max_risk_prob=0.5,
        slippage_bps=0.0,
        commission_bps=0.0,
        stamp_duty_bps=0.0,
        market_exposure_by_date={"2024-01-02": 0.5},
    )

    full_buy = full["orders"][full["orders"]["side"] == "buy"].iloc[0]
    half_buy = half["orders"][half["orders"]["side"] == "buy"].iloc[0]
    assert half_buy["value"] == pytest.approx(full_buy["value"] * 0.5)
