from __future__ import annotations

import pandas as pd


def format_stock_report(
    *,
    market: pd.Series,
    sector: pd.Series,
    stock: pd.Series,
    window_states: dict[int, str],
    sequence: pd.Series,
) -> str:
    return "\n".join(
        [
            "【1. 全A市场结构】",
            f"当前全A状态：{_get(market, 'final_state')}",
            f"全A趋势背景：{_get(market, 'trend_background')}",
            f"全A量价特点：{_get(market, 'main_features')}",
            f"市场风险偏好：{_get(market, 'risk_flags') or '未见突出风险'}",
            f"结论：{_get(market, 'final_state')}",
            "",
            "【2. 所属板块结构】",
            f"板块名称：{_get(sector, 'scope_id')}",
            f"板块趋势：{_get(sector, 'trend_background')}",
            f"板块量价结构：{_get(sector, 'main_features')}",
            f"板块相对全A强弱：{_get(sector, 'relative_strength_score')}",
            "板块内部共振程度：见板块评分",
            f"结论：{_get(sector, 'main_features')}",
            "",
            "【3. 个股多窗口结构】",
            f"股票代码：{_get(stock, 'scope_id')}",
            f"10日量价状态：{window_states.get(10, 'UNKNOWN')}",
            f"20日量价状态：{window_states.get(20, 'UNKNOWN')}",
            f"30日量价状态：{window_states.get(30, 'UNKNOWN')}",
            f"60日趋势背景：{window_states.get(60, 'UNKNOWN')}",
            f"120/240日大级别位置：{window_states.get(120, 'UNKNOWN')} / {window_states.get(240, 'UNKNOWN')}",
            "",
            "【4. 多日标签序列】",
            f"主要正常量价行为：{_get(sequence, 'normal_count')}",
            f"主要异常量价行为：{_get(sequence, 'abnormal_count')}",
            f"承接型标签数量：{_get(sequence, 'support_label_count')}",
            f"供应型标签数量：{_get(sequence, 'supply_label_count')}",
            f"最近半窗口 vs 前半窗口变化：{_get(sequence, 'bull_score_change')}",
            "",
            "【5. 综合结构判断】",
            f"当前阶段：{_get(stock, 'final_state')}",
            f"量价特点：{_get(stock, 'main_features')}",
            f"多空强弱：{_get(stock, 'confidence')}",
            f"是否与全A/板块共振：{_get(stock, 'resonance_score')}",
            "",
            "【6. 后续确认】",
            f"看强确认：{_get(stock, 'bullish_confirm_condition')}",
            f"看弱否定：{_get(stock, 'bearish_invalidate_condition')}",
            "需要继续观察：后续多窗口标签是否延续",
            "",
            "【7. 最终评级】",
            f"市场评分：{_get(stock, 'market_score')}",
            f"板块评分：{_get(stock, 'sector_score')}",
            f"个股评分：{_get(stock, 'self_score')}",
            f"共振评分：{_get(stock, 'resonance_score')}",
            f"最终评级：{_get(stock, 'final_rating')}",
            f"置信度：{_get(stock, 'confidence')}",
        ]
    )


def _get(series: pd.Series, key: str) -> object:
    value = series.get(key, "")
    if pd.isna(value):
        return ""
    return value
