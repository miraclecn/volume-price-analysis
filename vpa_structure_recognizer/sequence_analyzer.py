from __future__ import annotations

import pandas as pd


SEQUENCE_COLUMNS = [
    "date",
    "scope_type",
    "scope_id",
    "window_n",
    "parent_window_n",
    "normal_count",
    "abnormal_count",
    "abnormal_ratio",
    "bullish_label_count",
    "bearish_label_count",
    "neutral_label_count",
    "support_label_count",
    "supply_label_count",
    "high_volume_up_count",
    "high_volume_down_count",
    "high_volume_stall_count",
    "long_upper_shadow_count",
    "long_lower_shadow_count",
    "low_volume_pullback_count",
    "low_volume_rebound_count",
    "breakout_like_count",
    "breakdown_like_count",
    "last_part_bull_score",
    "previous_part_bull_score",
    "bull_score_change",
    "sequence_pattern",
    "sequence_strength_score",
]

BULLISH_LABELS = {
    "NORMAL_UP_CONFIRM",
    "HIGH_VOLUME_LOWER_SUPPORT",
    "BREAKDOWN_RECOVERY",
    "LOW_VOLUME_BIG_UP",
}
BEARISH_LABELS = {
    "NORMAL_DOWN_CONFIRM",
    "HIGH_VOLUME_UPPER_SUPPLY",
    "HIGH_VOLUME_LOW_PROGRESS",
    "LOW_VOLUME_BIG_DOWN",
    "BREAKOUT_PULLBACK",
}
SUPPORT_LABELS = {"HIGH_VOLUME_LOWER_SUPPORT", "BREAKDOWN_RECOVERY"}
SUPPLY_LABELS = {"HIGH_VOLUME_UPPER_SUPPLY", "BREAKOUT_PULLBACK"}


def analyze_sequences(labels: pd.DataFrame, trend_context: pd.DataFrame) -> pd.DataFrame:
    context_index = {
        (row.date, row.scope_type, row.scope_id, int(row.window_n)): row
        for row in trend_context.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []

    grouped = labels.sort_values(["scope_type", "scope_id", "window_n", "date"]).groupby(
        ["scope_type", "scope_id", "window_n"], sort=False
    )
    for (_scope_type, _scope_id, window_n), group in grouped:
        window = int(window_n)
        ordered = group.reset_index(drop=True)
        for idx, latest in ordered.iterrows():
            sequence = ordered.iloc[max(0, idx - window + 1) : idx + 1]
            context = context_index.get(
                (latest["date"], latest["scope_type"], latest["scope_id"], window)
            )
            rows.append(_stats_row(sequence, latest, context, window))

    return pd.DataFrame(rows, columns=SEQUENCE_COLUMNS)


def _stats_row(
    sequence: pd.DataFrame,
    latest: pd.Series,
    context: object | None,
    window_n: int,
) -> dict[str, object]:
    raw_labels = sequence["raw_label"]
    normal_count = int((sequence["normal_or_abnormal"] == "NORMAL").sum())
    abnormal_count = int((sequence["normal_or_abnormal"] == "ABNORMAL").sum())
    support_count = int(raw_labels.isin(SUPPORT_LABELS).sum())
    supply_count = int(raw_labels.isin(SUPPLY_LABELS).sum())
    high_volume_stall_count = int((raw_labels == "HIGH_VOLUME_LOW_PROGRESS").sum())
    low_volume_pullback_count = int((raw_labels == "LOW_VOLUME_BIG_DOWN").sum())
    low_volume_rebound_count = int((raw_labels == "LOW_VOLUME_BIG_UP").sum())
    breakout_like_count = int((raw_labels == "BREAKOUT_PULLBACK").sum())
    breakdown_like_count = int((raw_labels == "BREAKDOWN_RECOVERY").sum())
    previous_bull, last_bull = _split_bull_scores(sequence)
    bull_change = last_bull - previous_bull
    trend_label = getattr(context, "trend_label", "UNKNOWN")
    position_label = getattr(context, "position_label", "UNKNOWN")
    pattern = _classify_pattern(
        trend_label=trend_label,
        position_label=position_label,
        support_count=support_count,
        supply_count=supply_count,
        high_volume_stall_count=high_volume_stall_count,
        low_volume_pullback_count=low_volume_pullback_count,
        low_volume_rebound_count=low_volume_rebound_count,
        breakout_like_count=breakout_like_count,
        bull_score_change=bull_change,
        raw_labels=raw_labels,
    )

    return {
        "date": latest["date"],
        "scope_type": latest["scope_type"],
        "scope_id": latest["scope_id"],
        "window_n": window_n,
        "parent_window_n": latest["parent_window_n"],
        "normal_count": normal_count,
        "abnormal_count": abnormal_count,
        "abnormal_ratio": abnormal_count / len(sequence),
        "bullish_label_count": int(raw_labels.isin(BULLISH_LABELS).sum()),
        "bearish_label_count": int(raw_labels.isin(BEARISH_LABELS).sum()),
        "neutral_label_count": int((raw_labels == "NEUTRAL").sum()),
        "support_label_count": support_count,
        "supply_label_count": supply_count,
        "high_volume_up_count": int((raw_labels == "NORMAL_UP_CONFIRM").sum()),
        "high_volume_down_count": int((raw_labels == "NORMAL_DOWN_CONFIRM").sum()),
        "high_volume_stall_count": high_volume_stall_count,
        "long_upper_shadow_count": supply_count,
        "long_lower_shadow_count": support_count,
        "low_volume_pullback_count": low_volume_pullback_count,
        "low_volume_rebound_count": low_volume_rebound_count,
        "breakout_like_count": breakout_like_count,
        "breakdown_like_count": breakdown_like_count,
        "last_part_bull_score": last_bull,
        "previous_part_bull_score": previous_bull,
        "bull_score_change": bull_change,
        "sequence_pattern": pattern,
        "sequence_strength_score": _sequence_strength(pattern, bull_change, abnormal_count),
    }


def _split_bull_scores(sequence: pd.DataFrame) -> tuple[float, float]:
    half = max(1, len(sequence) // 2)
    previous = sequence.iloc[:half]["bull_bear_score"].mean()
    last = sequence.iloc[half:]["bull_bear_score"].mean()
    if pd.isna(last):
        last = previous
    return float(previous), float(last)


def _classify_pattern(
    *,
    trend_label: str,
    position_label: str,
    support_count: int,
    supply_count: int,
    high_volume_stall_count: int,
    low_volume_pullback_count: int,
    low_volume_rebound_count: int,
    breakout_like_count: int,
    bull_score_change: float,
    raw_labels: pd.Series,
) -> str:
    low_position = position_label in {"LOW", "MID_LOW"}
    high_position = position_label in {"MID_HIGH", "HIGH"}
    if (
        trend_label == "DOWNTREND"
        and low_position
        and support_count >= 1
        and bull_score_change > 0
    ):
        return "DECLINE_EXHAUSTION_PATTERN"
    if low_position and support_count >= 2 and supply_count == 0:
        return "LOW_LEVEL_SUPPORT_PATTERN"
    if (
        position_label == "HIGH"
        and supply_count >= 2
        and high_volume_stall_count >= 1
        and (low_volume_rebound_count >= 1 or trend_label == "WEAKENING")
    ):
        return "POSSIBLE_DISTRIBUTION_PATTERN"
    if breakout_like_count >= 1 and supply_count >= 2 and low_volume_rebound_count >= 1:
        return "FALSE_BREAKOUT_PATTERN"
    if high_position and (supply_count >= 2 or high_volume_stall_count >= 1):
        return "HIGH_LEVEL_SUPPLY_PATTERN"
    if (
        trend_label in {"UPTREND", "RECOVERING"}
        and int((raw_labels == "NORMAL_UP_CONFIRM").sum()) >= 2
        and supply_count == 0
    ):
        return "HEALTHY_UPTREND_PATTERN"
    return "NO_CLEAR_PATTERN"


def _sequence_strength(pattern: str, bull_score_change: float, abnormal_count: int) -> float:
    if pattern == "NO_CLEAR_PATTERN":
        return 0.0
    return round(min(100.0, 50.0 + abs(bull_score_change) + abnormal_count * 5.0), 4)
