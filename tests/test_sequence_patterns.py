import pandas as pd
import pytest

from vpa_structure_recognizer.sequence_analyzer import analyze_sequences


def _labels(raw_labels, bull_scores=None):
    bull_scores = bull_scores or [0.0] * len(raw_labels)
    return pd.DataFrame(
        [
            {
                "date": f"2024-01-{idx + 1:02d}",
                "scope_type": "stock",
                "scope_id": "000001.SZ",
                "window_n": len(raw_labels),
                "parent_window_n": 60,
                "raw_label": raw_label,
                "normal_or_abnormal": "ABNORMAL"
                if raw_label not in {"NORMAL_UP_CONFIRM", "LOW_VOLUME_SMALL_MOVE"}
                else "NORMAL",
                "volume_level": "HIGH_VOLUME"
                if raw_label.startswith("HIGH_VOLUME") or raw_label.startswith("NORMAL_")
                else "LOW_VOLUME",
                "bull_bear_score": bull_scores[idx],
                "supply_score": 80.0 if "SUPPLY" in raw_label or "PULLBACK" in raw_label else 0.0,
                "demand_score": 80.0 if "SUPPORT" in raw_label or raw_label == "NORMAL_UP_CONFIRM" else 0.0,
            }
            for idx, raw_label in enumerate(raw_labels)
        ]
    )


def _context(trend_label, position_label, window_n):
    return pd.DataFrame(
        [
            {
                "date": f"2024-01-{window_n:02d}",
                "scope_type": "stock",
                "scope_id": "000001.SZ",
                "window_n": window_n,
                "parent_window_n": 60,
                "trend_label": trend_label,
                "position_label": position_label,
            }
        ]
    )


@pytest.mark.parametrize(
    ("raw_labels", "bull_scores", "trend", "position", "expected"),
    [
        (
            [
                "NORMAL_DOWN_CONFIRM",
                "NORMAL_DOWN_CONFIRM",
                "LOW_VOLUME_BIG_DOWN",
                "HIGH_VOLUME_LOWER_SUPPORT",
            ],
            [-70, -60, -20, 30],
            "DOWNTREND",
            "LOW",
            "DECLINE_EXHAUSTION_PATTERN",
        ),
        (
            [
                "HIGH_VOLUME_LOWER_SUPPORT",
                "LOW_VOLUME_BIG_DOWN",
                "BREAKDOWN_RECOVERY",
                "LOW_VOLUME_SMALL_MOVE",
            ],
            [10, -10, 20, 5],
            "SIDEWAYS",
            "MID_LOW",
            "LOW_LEVEL_SUPPORT_PATTERN",
        ),
        (
            [
                "NORMAL_UP_CONFIRM",
                "LOW_VOLUME_SMALL_MOVE",
                "NORMAL_UP_CONFIRM",
                "LOW_VOLUME_SMALL_MOVE",
            ],
            [60, 10, 65, 5],
            "UPTREND",
            "MID",
            "HEALTHY_UPTREND_PATTERN",
        ),
        (
            [
                "NORMAL_UP_CONFIRM",
                "HIGH_VOLUME_UPPER_SUPPLY",
                "HIGH_VOLUME_LOW_PROGRESS",
                "BREAKOUT_PULLBACK",
            ],
            [60, -20, -10, -40],
            "UPTREND",
            "HIGH",
            "HIGH_LEVEL_SUPPLY_PATTERN",
        ),
        (
            [
                "HIGH_VOLUME_LOW_PROGRESS",
                "HIGH_VOLUME_UPPER_SUPPLY",
                "LOW_VOLUME_BIG_UP",
                "BREAKOUT_PULLBACK",
            ],
            [-5, -20, 20, -50],
            "WEAKENING",
            "HIGH",
            "POSSIBLE_DISTRIBUTION_PATTERN",
        ),
        (
            [
                "NORMAL_UP_CONFIRM",
                "BREAKOUT_PULLBACK",
                "LOW_VOLUME_BIG_UP",
                "HIGH_VOLUME_UPPER_SUPPLY",
            ],
            [50, -30, 15, -40],
            "SIDEWAYS",
            "MID_HIGH",
            "FALSE_BREAKOUT_PATTERN",
        ),
    ],
)
def test_sequence_patterns_are_identified(raw_labels, bull_scores, trend, position, expected):
    labels = _labels(raw_labels, bull_scores)
    context = _context(trend, position, len(raw_labels))

    stats = analyze_sequences(labels, context)
    latest = stats.iloc[-1]

    assert latest["sequence_pattern"] == expected
    assert latest["normal_count"] + latest["abnormal_count"] == len(raw_labels)
    assert latest["bull_score_change"] == pytest.approx(
        sum(bull_scores[len(bull_scores) // 2 :]) / 2
        - sum(bull_scores[: len(bull_scores) // 2]) / 2
    )
