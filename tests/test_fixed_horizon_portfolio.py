from __future__ import annotations

import pandas as pd

from ml_stock_selector.portfolio.constraints import FixedHorizonRiskFilterConfig
from ml_stock_selector.portfolio.fixed_horizon import construct_fixed_5d_risk_filter_targets
from ml_stock_selector.portfolio.holding_policy import HoldingState


def _constraints(**overrides):
    values = {
        "target_positions": 2,
        "hard_max_positions": 2,
        "max_initial_entries": 2,
        "max_new_entries_per_day": 2,
        "min_abs_rank_pct": 0.70,
        "risk_entry_max_rank_pct": 0.55,
        "risk_exit_rank_pct": 0.85,
        "min_adv20_amount": 50_000_000,
        "min_position_weight": 0.06,
        "max_position_weight": 0.50,
    }
    values.update(overrides)
    return FixedHorizonRiskFilterConfig(**values)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "code": "000001.SZ",
                "absolute_rank_pct": 0.95,
                "risk_rank_pct": 0.10,
                "is_bse": False,
                "is_st": False,
                "is_paused": False,
                "can_buy_next_open": True,
                "can_sell_next_open": True,
                "adv20_amount": 100_000_000,
            },
            {
                "trade_date": "2024-01-02",
                "code": "000002.SZ",
                "absolute_rank_pct": 0.80,
                "risk_rank_pct": 0.70,
                "is_bse": False,
                "is_st": False,
                "is_paused": False,
                "can_buy_next_open": True,
                "can_sell_next_open": True,
                "adv20_amount": 100_000_000,
            },
            {
                "trade_date": "2024-01-02",
                "code": "920001.BJ",
                "absolute_rank_pct": 0.99,
                "risk_rank_pct": 0.10,
                "is_bse": True,
                "is_st": False,
                "is_paused": False,
                "can_buy_next_open": True,
                "can_sell_next_open": True,
                "adv20_amount": 100_000_000,
            },
        ]
    )


def test_fixed_horizon_buys_absolute_rank_candidates_after_hard_and_risk_filters():
    result = construct_fixed_5d_risk_filter_targets(_candidates(), [], _constraints(), "2024-01-02")

    targets = result.targets
    assert targets["code"].tolist() == ["000001.SZ"]
    assert targets.iloc[0]["signal_action"] == "buy"
    assert targets.iloc[0]["entry_reason"] == "fixed_5d_abs_rank"
    assert targets.iloc[0]["target_weight"] == 0.50
    assert result.diagnostics.iloc[0]["risk_entry_rejected_count"] == 1


def test_fixed_horizon_retains_holding_that_drops_out_before_day_five():
    holding = HoldingState("000001.SZ", "2024-01-03", 10.0, 100.0, 3, 3, 0.95, 0.20, "fixed_5d_abs_rank")
    low_score_today = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-08",
                "code": "000001.SZ",
                "absolute_rank_pct": 0.01,
                "risk_rank_pct": 0.20,
                "can_sell_next_open": True,
            }
        ]
    )

    result = construct_fixed_5d_risk_filter_targets(low_score_today, [holding], _constraints(), "2024-01-08")

    held = result.targets.iloc[0]
    assert held["signal_action"] == "hold"
    assert held["exit_reason"] is None
    assert held["hold_reason"] == "fixed_horizon_not_due"


def test_fixed_horizon_only_time_or_risk_exit_and_respects_sell_block():
    holdings = [
        HoldingState("time", "2024-01-03", 10.0, 100.0, 5, 5, 0.90, 0.20, "fixed_5d_abs_rank"),
        HoldingState("risk", "2024-01-05", 10.0, 100.0, 2, 2, 0.90, 0.20, "fixed_5d_abs_rank"),
        HoldingState("blocked", "2024-01-03", 10.0, 100.0, 5, 5, 0.90, 0.20, "fixed_5d_abs_rank"),
    ]
    latest = pd.DataFrame(
        [
            {"trade_date": "2024-01-10", "code": "time", "absolute_rank_pct": 0.01, "risk_rank_pct": 0.20, "can_sell_next_open": True},
            {"trade_date": "2024-01-10", "code": "risk", "absolute_rank_pct": 0.95, "risk_rank_pct": 0.90, "can_sell_next_open": True},
            {"trade_date": "2024-01-10", "code": "blocked", "absolute_rank_pct": 0.01, "risk_rank_pct": 0.20, "can_sell_next_open": False},
        ]
    )

    result = construct_fixed_5d_risk_filter_targets(latest, holdings, _constraints(), "2024-01-10")
    by_code = result.targets.set_index("code")

    assert by_code.loc["time", "signal_action"] == "sell"
    assert by_code.loc["time", "exit_reason"] == "time_exit"
    assert by_code.loc["risk", "signal_action"] == "sell"
    assert by_code.loc["risk", "exit_reason"] == "risk_exit"
    assert by_code.loc["blocked", "signal_action"] == "sell_blocked"
    assert by_code.loc["blocked", "exit_reason"] == "time_exit"
    assert "score_exit" not in set(result.targets["exit_reason"].dropna())
    assert "not_candidate_after_target_days" not in set(result.targets["exit_reason"].dropna())
    assert "trailing_exit" not in set(result.targets["exit_reason"].dropna())


def test_fixed_horizon_renews_expired_holding_when_still_top_30_candidate():
    holding = HoldingState("held", "2024-01-03", 10.0, 100.0, 5, 5, 0.90, 0.20, "fixed_5d_abs_rank")
    rows = [
        {
            "trade_date": "2024-01-10",
            "code": f"rank_{idx:02d}",
            "absolute_rank_pct": 0.99 - idx * 0.001,
            "risk_rank_pct": 0.10,
            "is_bse": False,
            "is_st": False,
            "is_paused": False,
            "can_buy_next_open": True,
            "can_sell_next_open": True,
            "adv20_amount": 100_000_000,
        }
        for idx in range(29)
    ]
    rows.append(
        {
            "trade_date": "2024-01-10",
            "code": "held",
            "absolute_rank_pct": 0.80,
            "risk_rank_pct": 0.10,
            "is_bse": False,
            "is_st": False,
            "is_paused": False,
            "can_buy_next_open": True,
            "can_sell_next_open": True,
            "adv20_amount": 100_000_000,
        }
    )

    result = construct_fixed_5d_risk_filter_targets(pd.DataFrame(rows), [holding], _constraints(), "2024-01-10")
    held = result.targets[result.targets["code"] == "held"].iloc[0]

    assert held["signal_action"] == "hold"
    assert held["hold_reason"] == "renewed_top_candidate"
    assert held["exit_reason"] is None


def test_fixed_horizon_exits_expired_holding_when_candidate_rank_below_30():
    holding = HoldingState("held", "2024-01-03", 10.0, 100.0, 5, 5, 0.90, 0.20, "fixed_5d_abs_rank")
    rows = [
        {
            "trade_date": "2024-01-10",
            "code": f"rank_{idx:02d}",
            "absolute_rank_pct": 0.99 - idx * 0.001,
            "risk_rank_pct": 0.10,
            "is_bse": False,
            "is_st": False,
            "is_paused": False,
            "can_buy_next_open": True,
            "can_sell_next_open": True,
            "adv20_amount": 100_000_000,
        }
        for idx in range(30)
    ]
    rows.append(
        {
            "trade_date": "2024-01-10",
            "code": "held",
            "absolute_rank_pct": 0.70,
            "risk_rank_pct": 0.10,
            "is_bse": False,
            "is_st": False,
            "is_paused": False,
            "can_buy_next_open": True,
            "can_sell_next_open": True,
            "adv20_amount": 100_000_000,
        }
    )

    result = construct_fixed_5d_risk_filter_targets(pd.DataFrame(rows), [holding], _constraints(), "2024-01-10")
    held = result.targets[result.targets["code"] == "held"].iloc[0]

    assert held["signal_action"] == "sell"
    assert held["exit_reason"] == "time_exit"


def test_fixed_horizon_no_risk_exit_variant_disables_early_risk_sell():
    holding = HoldingState("risk", "2024-01-05", 10.0, 100.0, 2, 2, 0.90, 0.20, "fixed_5d_abs_rank")
    latest = pd.DataFrame(
        [{"trade_date": "2024-01-10", "code": "risk", "absolute_rank_pct": 0.95, "risk_rank_pct": 0.99, "can_sell_next_open": True}]
    )

    result = construct_fixed_5d_risk_filter_targets(
        latest,
        [holding],
        _constraints(strategy_id="abs_ranker_fixed_5d_no_risk_exit_v1", enable_risk_exit=False),
        "2024-01-10",
    )

    assert result.targets.iloc[0]["signal_action"] == "hold"
    assert result.targets.iloc[0]["exit_reason"] is None
