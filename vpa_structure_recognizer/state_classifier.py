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
    context_index = {
        (row.date, row.scope_type, row.scope_id, int(row.window_n)): row
        for row in trend_context.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []

    grouped = sequence_stats.groupby(["date", "scope_type", "scope_id"], sort=True)
    for (date, scope_type, scope_id), group in grouped:
        states_by_window: dict[int, str] = {}
        strengths: list[float] = []
        contexts = []
        for seq in group.itertuples(index=False):
            context = context_index.get((seq.date, seq.scope_type, seq.scope_id, int(seq.window_n)))
            contexts.append(context)
            state = _state_for_sequence(seq.sequence_pattern, context)
            states_by_window[int(seq.window_n)] = state
            if state != "UNCLEAR":
                strengths.append(float(seq.sequence_strength_score))

        final_state = _final_state(states_by_window, contexts)
        trend, position = _background(contexts)
        rows.append(
            {
                "date": date,
                "scope_type": scope_type,
                "scope_id": scope_id,
                "state_10": states_by_window.get(10),
                "state_20": states_by_window.get(20),
                "state_30": states_by_window.get(30),
                "state_60": states_by_window.get(60),
                "state_120": states_by_window.get(120),
                "state_240": states_by_window.get(240),
                "final_state": final_state,
                "trend_background": trend,
                "position_background": position,
                "market_score": None,
                "sector_score": None,
                "self_score": None,
                "relative_strength_score": None,
                "resonance_score": None,
                "final_rating": None,
                "confidence": _confidence(strengths, final_state),
                "main_features": _main_features(final_state),
                "risk_flags": _risk_flags(final_state),
                "bullish_confirm_condition": _bullish_condition(final_state),
                "bearish_invalidate_condition": _bearish_condition(final_state),
            }
        )

    output = pd.DataFrame(rows, columns=STATE_COLUMNS)
    return output.astype(object).where(pd.notna(output), None)


def _state_for_sequence(pattern: str, context: object | None) -> str:
    position = getattr(context, "position_label", "UNKNOWN")
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


def _final_state(states_by_window: dict[int, str], contexts: list[object | None]) -> str:
    states = list(states_by_window.values())
    positions = {getattr(context, "position_label", "UNKNOWN") for context in contexts if context}
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


def _background(contexts: list[object | None]) -> tuple[str, str]:
    known = [context for context in contexts if context is not None]
    if not known:
        return "UNKNOWN", "UNKNOWN"
    widest = max(known, key=lambda context: int(context.window_n))
    return widest.trend_label, widest.position_label


def _confidence(strengths: list[float], final_state: str) -> float:
    if final_state == "UNCLEAR" or not strengths:
        return 0.0
    return round(min(1.0, sum(strengths) / len(strengths) / 100.0), 4)


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
