from __future__ import annotations

import pandas as pd


STATE_COLUMNS = [
    "date",
    "scope_type",
    "scope_id",
    "state_10",
    "state_20",
    "state_30",
    "state_60",
    "state_120",
    "state_240",
    "final_state",
    "trend_background",
    "position_background",
    "market_score",
    "sector_score",
    "self_score",
    "relative_strength_score",
    "resonance_score",
    "final_rating",
    "confidence",
    "main_features",
    "risk_flags",
    "bullish_confirm_condition",
    "bearish_invalidate_condition",
]


def classify_structure_states(
    sequence_stats: pd.DataFrame,
    trend_context: pd.DataFrame,
) -> pd.DataFrame:
    context = trend_context[
        ["date", "scope_type", "scope_id", "window_n", "trend_label", "position_label"]
    ]
    merged = sequence_stats.merge(
        context,
        on=["date", "scope_type", "scope_id", "window_n"],
        how="left",
    )
    merged["window_state"] = [
        _state_for_pattern(pattern, position)
        for pattern, position in zip(
            merged["sequence_pattern"],
            merged["position_label"].fillna("UNKNOWN"),
        )
    ]
    index_columns = ["date", "scope_type", "scope_id"]
    state_values = merged[index_columns + ["window_n", "window_state"]].drop_duplicates(
        index_columns + ["window_n"], keep="last"
    )
    states_wide = (
        state_values.pivot(
            index=index_columns,
            columns="window_n",
            values="window_state",
        )
        .rename(columns=lambda window: f"state_{int(window)}")
        .reset_index()
    )
    for column in ["state_10", "state_20", "state_30", "state_60", "state_120", "state_240"]:
        if column not in states_wide.columns:
            states_wide[column] = None

    background = (
        merged.sort_values(index_columns + ["window_n"])
        .drop_duplicates(index_columns, keep="last")[
            index_columns + ["trend_label", "position_label"]
        ]
    ).rename(
        columns={
            "trend_label": "trend_background",
            "position_label": "position_background",
        }
    )
    output = states_wide.merge(background, on=index_columns, how="left")
    output["final_state"] = [
        _final_state_from_row(row)
        for row in output[
            ["state_10", "state_20", "state_30", "state_60", "state_120", "state_240", "position_background"]
        ].itertuples(index=False)
    ]
    confidence_source = merged[merged["window_state"] != "UNCLEAR"]
    confidence = (
        confidence_source.groupby(index_columns)["sequence_strength_score"].mean().div(100).fillna(0)
    )
    output = output.merge(confidence.rename("confidence"), on=index_columns, how="left")
    output["confidence"] = output["confidence"].where(output["final_state"] != "UNCLEAR", 0.0).round(4)
    output["market_score"] = None
    output["sector_score"] = None
    output["self_score"] = None
    output["relative_strength_score"] = None
    output["resonance_score"] = None
    output["final_rating"] = None
    output["main_features"] = output["final_state"].map(_main_features)
    output["risk_flags"] = output["final_state"].map(_risk_flags)
    output["bullish_confirm_condition"] = output["final_state"].map(_bullish_condition)
    output["bearish_invalidate_condition"] = output["final_state"].map(_bearish_condition)
    return output[STATE_COLUMNS]


def _state_for_pattern(pattern: str, position: str) -> str:
    if pattern == "DECLINE_EXHAUSTION_PATTERN":
        return "DECLINE_EXHAUSTION"
    if pattern == "LOW_LEVEL_SUPPORT_PATTERN":
        return "LOW_LEVEL_SUPPORT"
    if pattern == "HEALTHY_UPTREND_PATTERN":
        return "HEALTHY_UPTREND"
    if pattern == "HIGH_LEVEL_SUPPLY_PATTERN":
        return "HIGH_LEVEL_SUPPLY"
    if pattern == "POSSIBLE_DISTRIBUTION_PATTERN":
        if position in {"MID_HIGH", "HIGH"}:
            return "POSSIBLE_DISTRIBUTION"
        return "HIGH_LEVEL_SUPPLY"
    if pattern == "FALSE_BREAKOUT_PATTERN":
        return "BREAKOUT_ATTEMPT"
    return "UNCLEAR"


def _final_state_from_row(row: object) -> str:
    states = [value for value in row[:-1] if value is not None and not pd.isna(value)]
    positions = {row.position_background}
    if "POSSIBLE_DISTRIBUTION" in states:
        return "POSSIBLE_DISTRIBUTION"
    low_support_evidence = sum(
        state in {"DECLINE_EXHAUSTION", "LOW_LEVEL_SUPPORT"} for state in states
    )
    if low_support_evidence >= 2 and positions & {"LOW", "MID_LOW"}:
        return "POSSIBLE_ACCUMULATION"
    for candidate in [
        "HEALTHY_UPTREND",
        "HIGH_LEVEL_SUPPLY",
        "DECLINE_EXHAUSTION",
        "LOW_LEVEL_SUPPORT",
        "BREAKOUT_ATTEMPT",
    ]:
        if candidate in states:
            return candidate
    return "UNCLEAR"


def _main_features(final_state: str) -> str:
    mapping = {
        "POSSIBLE_ACCUMULATION": "低位承接增强，多窗口卖压衰竭",
        "HEALTHY_UPTREND": "上涨有量，回调相对温和",
        "DECLINE_EXHAUSTION": "下跌破坏力下降",
        "LOW_LEVEL_SUPPORT": "低位承接标签增加",
        "HIGH_LEVEL_SUPPLY": "高位供应增强，推进效率下降",
        "POSSIBLE_DISTRIBUTION": "高位供应占优，买盘推进困难",
        "BREAKOUT_ATTEMPT": "突破尝试后确认不足",
    }
    return mapping.get(final_state, "结构不明确")


def _risk_flags(final_state: str) -> str:
    if final_state == "POSSIBLE_DISTRIBUTION":
        return "高位供应风险"
    if final_state == "HIGH_LEVEL_SUPPLY":
        return "供应增强"
    if final_state == "BREAKOUT_ATTEMPT":
        return "突破失败风险"
    return ""


def _bullish_condition(final_state: str) -> str:
    if final_state in {"POSSIBLE_ACCUMULATION", "LOW_LEVEL_SUPPORT", "DECLINE_EXHAUSTION"}:
        return "后续放量上攻并站上短期高点"
    if final_state == "HEALTHY_UPTREND":
        return "回调缩量且趋势均线保持上行"
    return "等待多窗口需求确认"


def _bearish_condition(final_state: str) -> str:
    if final_state in {"POSSIBLE_DISTRIBUTION", "HIGH_LEVEL_SUPPLY"}:
        return "跌破短期箱体或放量长阴"
    return "跌破近期低点且承接标签消失"
