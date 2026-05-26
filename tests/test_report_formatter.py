import pandas as pd

from vpa_structure_recognizer.report_formatter import format_stock_report


def test_format_stock_report_contains_spec_sections():
    report = format_stock_report(
        market=pd.Series(
            {
                "final_state": "HEALTHY_UPTREND",
                "trend_background": "UPTREND",
                "main_features": "市场健康",
                "risk_flags": "",
            }
        ),
        sector=pd.Series(
            {
                "scope_id": "BK001",
                "trend_background": "UPTREND",
                "main_features": "板块共振",
                "relative_strength_score": 20,
            }
        ),
        stock=pd.Series(
            {
                "scope_id": "000001.SZ",
                "final_state": "HEALTHY_UPTREND",
                "final_rating": "A",
                "market_score": 90,
                "sector_score": 85,
                "self_score": 95,
                "resonance_score": 100,
                "confidence": 0.8,
                "main_features": "上涨有量",
                "bullish_confirm_condition": "缩量回调",
                "bearish_invalidate_condition": "放量破位",
            }
        ),
        window_states={20: "HEALTHY_UPTREND", 60: "UPTREND"},
        sequence=pd.Series(
            {
                "normal_count": 4,
                "abnormal_count": 1,
                "support_label_count": 1,
                "supply_label_count": 0,
                "bull_score_change": 20,
            }
        ),
    )

    for section in [
        "【1. 全A市场结构】",
        "【2. 所属板块结构】",
        "【3. 个股多窗口结构】",
        "【4. 多日标签序列】",
        "【5. 综合结构判断】",
        "【6. 后续确认】",
        "【7. 最终评级】",
    ]:
        assert section in report
    assert "000001.SZ" in report
