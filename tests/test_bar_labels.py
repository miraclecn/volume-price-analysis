import pandas as pd

from vpa_structure_recognizer.bar_labeler import classify_volume_level, label_bars


def _feature_row(**overrides):
    row = {
        "date": "2024-01-02",
        "scope_type": "stock",
        "scope_id": "000001.SZ",
        "window_n": 20,
        "ret_pct": 0.03,
        "range_pct": 0.08,
        "body_pct": 0.04,
        "upper_shadow_pct": 0.01,
        "lower_shadow_pct": 0.01,
        "body_ratio": 0.5,
        "upper_shadow_ratio": 0.2,
        "lower_shadow_ratio": 0.2,
        "close_position": 0.75,
        "vol_rvol_n": 1.2,
        "range_rvol_n": 1.0,
        "price_high_n": 11.0,
        "price_low_n": 9.0,
        "high": 10.8,
        "low": 9.8,
        "close": 10.6,
    }
    row.update(overrides)
    return row


def test_volume_level_thresholds_match_spec():
    assert classify_volume_level(0.69) == "LOW_VOLUME"
    assert classify_volume_level(1.0) == "NORMAL_VOLUME"
    assert classify_volume_level(1.5) == "MILD_HIGH_VOLUME"
    assert classify_volume_level(2.0) == "HIGH_VOLUME"
    assert classify_volume_level(2.6) == "EXTREME_HIGH_VOLUME"


def test_label_bars_marks_normal_up_confirm():
    labels = label_bars(pd.DataFrame([_feature_row()]), {20: [60]})
    row = labels.iloc[0]

    assert row["raw_label"] == "NORMAL_UP_CONFIRM"
    assert row["normal_or_abnormal"] == "NORMAL"
    assert row["volume_level"] == "MILD_HIGH_VOLUME"
    assert row["bull_bear_score"] > 0
    assert row["demand_score"] > row["supply_score"]


def test_label_bars_marks_high_volume_upper_supply_as_abnormal():
    labels = label_bars(
        pd.DataFrame(
            [
                _feature_row(
                    ret_pct=0.01,
                    vol_rvol_n=1.6,
                    upper_shadow_ratio=0.5,
                    close_position=0.55,
                    body_ratio=0.25,
                )
            ]
        ),
        {20: [60]},
    )
    row = labels.iloc[0]

    assert row["raw_label"] == "HIGH_VOLUME_UPPER_SUPPLY"
    assert row["normal_or_abnormal"] == "ABNORMAL"
    assert row["supply_score"] > row["demand_score"]


def test_single_day_labels_do_not_emit_stage_conclusions():
    stress_rows = pd.DataFrame(
        [
            _feature_row(vol_rvol_n=3.0, range_rvol_n=0.5, body_ratio=0.2),
            _feature_row(vol_rvol_n=0.6, ret_pct=0.05, range_rvol_n=1.3),
            _feature_row(vol_rvol_n=0.6, ret_pct=-0.05, range_rvol_n=1.3, close_position=0.2),
        ]
    )

    labels = label_bars(stress_rows, {20: [60]})

    assert not labels["raw_label"].isin(
        ["POSSIBLE_ACCUMULATION", "POSSIBLE_DISTRIBUTION"]
    ).any()
