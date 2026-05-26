import pandas as pd

from vpa_structure_recognizer.state_classifier import classify_structure_states


def _sequence_row(window_n, pattern, strength=80.0):
    return {
        "date": "2024-01-31",
        "scope_type": "stock",
        "scope_id": "000001.SZ",
        "window_n": window_n,
        "parent_window_n": 60,
        "sequence_pattern": pattern,
        "sequence_strength_score": strength,
        "bull_score_change": 20.0,
        "support_label_count": 2 if "SUPPORT" in pattern or "EXHAUSTION" in pattern else 0,
        "supply_label_count": 2 if "SUPPLY" in pattern or "DISTRIBUTION" in pattern else 0,
    }


def _context_row(window_n, trend="DOWNTREND", position="LOW"):
    return {
        "date": "2024-01-31",
        "scope_type": "stock",
        "scope_id": "000001.SZ",
        "window_n": window_n,
        "trend_label": trend,
        "position_label": position,
    }


def test_possible_accumulation_requires_multi_window_low_position_evidence():
    sequences = pd.DataFrame(
        [
            _sequence_row(20, "DECLINE_EXHAUSTION_PATTERN"),
            _sequence_row(30, "LOW_LEVEL_SUPPORT_PATTERN"),
        ]
    )
    context = pd.DataFrame([_context_row(20), _context_row(30)])

    states = classify_structure_states(sequences, context)
    row = states.iloc[0]

    assert row["state_20"] == "DECLINE_EXHAUSTION"
    assert row["state_30"] == "LOW_LEVEL_SUPPORT"
    assert row["final_state"] == "POSSIBLE_ACCUMULATION"
    assert row["confidence"] > 0
    assert "承接" in row["main_features"]


def test_possible_distribution_requires_high_position_context():
    sequences = pd.DataFrame([_sequence_row(20, "POSSIBLE_DISTRIBUTION_PATTERN")])
    mid_context = pd.DataFrame([_context_row(20, trend="SIDEWAYS", position="MID")])
    high_context = pd.DataFrame([_context_row(20, trend="WEAKENING", position="HIGH")])

    mid_state = classify_structure_states(sequences, mid_context).iloc[0]
    high_state = classify_structure_states(sequences, high_context).iloc[0]

    assert mid_state["final_state"] != "POSSIBLE_DISTRIBUTION"
    assert high_state["final_state"] == "POSSIBLE_DISTRIBUTION"
    assert "供应" in high_state["risk_flags"]


def test_healthy_uptrend_pattern_maps_to_final_state():
    sequences = pd.DataFrame([_sequence_row(20, "HEALTHY_UPTREND_PATTERN")])
    context = pd.DataFrame([_context_row(20, trend="UPTREND", position="MID")])

    state = classify_structure_states(sequences, context).iloc[0]

    assert state["final_state"] == "HEALTHY_UPTREND"
    assert state["trend_background"] == "UPTREND"
    assert state["position_background"] == "MID"
