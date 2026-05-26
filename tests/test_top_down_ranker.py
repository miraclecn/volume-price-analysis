import pandas as pd

from vpa_structure_recognizer.top_down_ranker import rank_top_down


def _state(scope_type, scope_id, final_state, confidence=0.8):
    return {
        "date": "2024-01-31",
        "scope_type": scope_type,
        "scope_id": scope_id,
        "final_state": final_state,
        "confidence": confidence,
        "risk_flags": "",
        "main_features": final_state,
    }


def test_top_down_ranking_rewards_three_level_resonance():
    states = pd.DataFrame(
        [
            _state("market", "ALL_A", "HEALTHY_UPTREND"),
            _state("sector", "BK001", "HEALTHY_UPTREND"),
            _state("stock", "000001.SZ", "HEALTHY_UPTREND"),
        ]
    )

    ranked = rank_top_down(states, {"000001.SZ": "BK001"})
    stock = ranked[ranked["scope_type"] == "stock"].iloc[0]

    assert stock["market_score"] >= 80
    assert stock["sector_score"] >= 80
    assert stock["self_score"] >= 80
    assert stock["resonance_score"] == 100
    assert stock["final_rating"] == "A"


def test_weak_market_downgrades_stock_rating():
    states = pd.DataFrame(
        [
            _state("market", "ALL_A", "POSSIBLE_DISTRIBUTION"),
            _state("sector", "BK001", "HEALTHY_UPTREND"),
            _state("stock", "000001.SZ", "HEALTHY_UPTREND"),
        ]
    )

    ranked = rank_top_down(states, {"000001.SZ": "BK001"})
    stock = ranked[ranked["scope_type"] == "stock"].iloc[0]

    assert stock["market_score"] < 40
    assert stock["final_rating"] == "C"


def test_strong_stock_in_weak_sector_is_watch_only():
    states = pd.DataFrame(
        [
            _state("market", "ALL_A", "HEALTHY_UPTREND"),
            _state("sector", "BK001", "HIGH_LEVEL_SUPPLY"),
            _state("stock", "000001.SZ", "HEALTHY_UPTREND"),
        ]
    )

    ranked = rank_top_down(states, {"000001.SZ": "BK001"})
    stock = ranked[ranked["scope_type"] == "stock"].iloc[0]

    assert stock["sector_score"] < 40
    assert stock["final_rating"] == "C"
    assert "逆势个股" in stock["risk_flags"]
