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
    stats = labels.sort_values(["scope_type", "scope_id", "window_n", "date"]).reset_index(
        drop=True
    )
    stats["_normal"] = (stats["normal_or_abnormal"] == "NORMAL").astype(int)
    stats["_abnormal"] = (stats["normal_or_abnormal"] == "ABNORMAL").astype(int)
    stats["_bullish"] = stats["raw_label"].isin(BULLISH_LABELS).astype(int)
    stats["_bearish"] = stats["raw_label"].isin(BEARISH_LABELS).astype(int)
    stats["_neutral"] = (stats["raw_label"] == "NEUTRAL").astype(int)
    stats["_support"] = stats["raw_label"].isin(SUPPORT_LABELS).astype(int)
    stats["_supply"] = stats["raw_label"].isin(SUPPLY_LABELS).astype(int)
    stats["_high_volume_up"] = (stats["raw_label"] == "NORMAL_UP_CONFIRM").astype(int)
    stats["_high_volume_down"] = (stats["raw_label"] == "NORMAL_DOWN_CONFIRM").astype(int)
    stats["_high_volume_stall"] = (stats["raw_label"] == "HIGH_VOLUME_LOW_PROGRESS").astype(int)
    stats["_low_volume_pullback"] = (stats["raw_label"] == "LOW_VOLUME_BIG_DOWN").astype(int)
    stats["_low_volume_rebound"] = (stats["raw_label"] == "LOW_VOLUME_BIG_UP").astype(int)
    stats["_breakout_like"] = (stats["raw_label"] == "BREAKOUT_PULLBACK").astype(int)
    stats["_breakdown_like"] = (stats["raw_label"] == "BREAKDOWN_RECOVERY").astype(int)

    pieces = [
        _rolling_stats_for_window(group, int(window_n))
        for window_n, group in stats.groupby("window_n", sort=False)
    ]
    rolled = pd.concat(pieces, ignore_index=True)
    context = trend_context[
        ["date", "scope_type", "scope_id", "window_n", "trend_label", "position_label"]
    ]
    rolled = rolled.merge(
        context,
        on=["date", "scope_type", "scope_id", "window_n"],
        how="left",
    )
    rolled["sequence_pattern"] = [
        _classify_pattern(
            trend_label=row.trend_label if pd.notna(row.trend_label) else "UNKNOWN",
            position_label=row.position_label if pd.notna(row.position_label) else "UNKNOWN",
            support_count=int(row.support_label_count),
            supply_count=int(row.supply_label_count),
            high_volume_stall_count=int(row.high_volume_stall_count),
            low_volume_pullback_count=int(row.low_volume_pullback_count),
            low_volume_rebound_count=int(row.low_volume_rebound_count),
            breakout_like_count=int(row.breakout_like_count),
            bull_score_change=float(row.bull_score_change),
            normal_up_count=int(row.high_volume_up_count),
        )
        for row in rolled.itertuples(index=False)
    ]
    rolled["sequence_strength_score"] = [
        _sequence_strength(pattern, change, abnormal)
        for pattern, change, abnormal in zip(
            rolled["sequence_pattern"],
            rolled["bull_score_change"],
            rolled["abnormal_count"],
        )
    ]
    return rolled[SEQUENCE_COLUMNS]


def _rolling_stats_for_window(group: pd.DataFrame, window_n: int) -> pd.DataFrame:
    output = group[
        ["date", "scope_type", "scope_id", "window_n", "parent_window_n"]
    ].copy()
    grouped = group.groupby(["scope_type", "scope_id"], sort=False)
    position = grouped.cumcount() + 1
    count = position.clip(upper=window_n)
    for source, target in [
        ("_normal", "normal_count"),
        ("_abnormal", "abnormal_count"),
        ("_bullish", "bullish_label_count"),
        ("_bearish", "bearish_label_count"),
        ("_neutral", "neutral_label_count"),
        ("_support", "support_label_count"),
        ("_supply", "supply_label_count"),
        ("_high_volume_up", "high_volume_up_count"),
        ("_high_volume_down", "high_volume_down_count"),
        ("_high_volume_stall", "high_volume_stall_count"),
        ("_low_volume_pullback", "low_volume_pullback_count"),
        ("_low_volume_rebound", "low_volume_rebound_count"),
        ("_breakout_like", "breakout_like_count"),
        ("_breakdown_like", "breakdown_like_count"),
    ]:
        output[target] = _rolling_sum(group, source, window_n).fillna(0).astype(int)
    output["abnormal_ratio"] = output["abnormal_count"] / count
    output["long_upper_shadow_count"] = output["supply_label_count"]
    output["long_lower_shadow_count"] = output["support_label_count"]
    half = max(1, window_n // 2)
    last, previous = _half_window_means(group, half)
    output["last_part_bull_score"] = last
    output["previous_part_bull_score"] = previous
    output["bull_score_change"] = output["last_part_bull_score"] - output[
        "previous_part_bull_score"
    ]
    return output


def _rolling_sum(group: pd.DataFrame, column: str, window: int) -> pd.Series:
    keys = [group["scope_type"], group["scope_id"]]
    source = group[column].fillna(0)
    cumsum = source.groupby(keys, sort=False).cumsum()
    shifted = cumsum.groupby(keys, sort=False).shift(window).fillna(0)
    return cumsum - shifted


def _half_window_means(group: pd.DataFrame, half: int) -> tuple[pd.Series, pd.Series]:
    keys = [group["scope_type"], group["scope_id"]]
    grouped = group.groupby(["scope_type", "scope_id"], sort=False)
    position = grouped.cumcount() + 1
    cumsum = grouped["bull_bear_score"].cumsum()

    last_start = cumsum.groupby(keys, sort=False).shift(half).fillna(0)
    last_sum = cumsum - last_start
    last_count = position.clip(upper=half)
    last = last_sum / last_count

    previous_end = last_start
    previous_start = cumsum.groupby(keys, sort=False).shift(half * 2).fillna(0)
    previous_count = (position - half).clip(lower=0, upper=half)
    previous_sum = previous_end - previous_start
    previous = previous_sum.where(previous_count > 0, last) / previous_count.where(
        previous_count > 0, 1
    )
    return last, previous


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
    normal_up_count: int,
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
        and normal_up_count >= 2
        and supply_count == 0
    ):
        return "HEALTHY_UPTREND_PATTERN"
    return "NO_CLEAR_PATTERN"


def _sequence_strength(pattern: str, bull_score_change: float, abnormal_count: int) -> float:
    if pattern == "NO_CLEAR_PATTERN":
        return 0.0
    return round(min(100.0, 50.0 + abs(bull_score_change) + abnormal_count * 5.0), 4)
