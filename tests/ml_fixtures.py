from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from vpa_structure_recognizer.storage import init_vpa_db, upsert_dataframe


def normalized_bars() -> pd.DataFrame:
    rows = []
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    for code_idx, code in enumerate(["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]):
        base = 10.0 + code_idx
        for i, date in enumerate(dates):
            close = base + i * (0.2 + code_idx * 0.05)
            rows.append(
                {
                    "trade_date": date,
                    "code": code,
                    "open": close - 0.05,
                    "high": close + 0.20,
                    "low": close - 0.25,
                    "close": close,
                    "prev_close": base + max(i - 1, 0) * (0.2 + code_idx * 0.05),
                    "volume": 1000 + code_idx * 100 + i * 10,
                    "amount": (1000 + code_idx * 100 + i * 10) * close,
                    "turnover_rate": 1.0 + code_idx * 0.1,
                    "is_st": code == "000004.SZ" and i == 1,
                    "is_paused": code == "000003.SZ" and i == 2,
                    "limit_up": close + 1.0,
                    "limit_down": close - 1.0,
                    "industry_code": "I1" if code_idx < 2 else "I2",
                    "industry_name": "Industry 1" if code_idx < 2 else "Industry 2",
                }
            )
    return pd.DataFrame(rows)


def create_alpha_data_db(path: Path, frame: pd.DataFrame | None = None) -> Path:
    data = normalized_bars() if frame is None else frame
    con = duckdb.connect(str(path))
    con.register("bars", data)
    con.execute("create table stock_bar_normalized_daily as select * from bars")
    con.unregister("bars")
    con.close()
    return path


def create_vpa_db(path: Path) -> Path:
    con = init_vpa_db(path)
    bars = normalized_bars()
    features = []
    labels = []
    sequences = []
    states = []
    for row in bars.itertuples(index=False):
        for window in [5, 20]:
            features.append(
                {
                    "date": row.trade_date,
                    "scope_type": "stock",
                    "scope_id": row.code,
                    "window_n": window,
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "prev_close": row.prev_close,
                    "volume": row.volume,
                    "amount": row.amount,
                    "ret_pct": row.close / row.prev_close - 1.0 if row.prev_close else 0.0,
                    "range_pct": (row.high - row.low) / row.close,
                    "body_pct": (row.close - row.open) / row.open,
                    "upper_shadow_pct": (row.high - row.close) / row.close,
                    "lower_shadow_pct": (row.open - row.low) / row.open,
                    "body_ratio": 0.2,
                    "upper_shadow_ratio": 0.4,
                    "lower_shadow_ratio": 0.4,
                    "close_position": 0.6,
                    "vol_ma_n": row.volume,
                    "vol_rvol_n": 1.1,
                    "range_pct_ma_n": 0.03,
                    "range_rvol_n": 1.2,
                    "body_pct_ma_n": 0.01,
                    "body_rvol_n": 0.9,
                    "price_high_n": row.high,
                    "price_low_n": row.low,
                    "price_position_n": 0.7,
                    "ma_n": row.close,
                    "ma_slope_n": 0.01,
                }
            )
            labels.append(
                {
                    "date": row.trade_date,
                    "scope_type": "stock",
                    "scope_id": row.code,
                    "window_n": window,
                    "parent_window_n": 20 if window == 5 else 60,
                    "raw_label": "NORMAL_UP_CONFIRM",
                    "normal_or_abnormal": "NORMAL",
                    "volume_level": "NORMAL",
                    "price_result_level": "UP",
                    "efficiency_level": "GOOD",
                    "bull_bear_score": 0.6,
                    "supply_score": 0.2,
                    "demand_score": 0.7,
                    "volatility_score": 0.3,
                    "description": "fixture",
                }
            )
            sequences.append(
                {
                    "date": row.trade_date,
                    "scope_type": "stock",
                    "scope_id": row.code,
                    "window_n": window,
                    "parent_window_n": 20 if window == 5 else 60,
                    "normal_count": 3,
                    "abnormal_count": 1,
                    "abnormal_ratio": 0.25,
                    "bullish_label_count": 2,
                    "bearish_label_count": 0,
                    "neutral_label_count": 1,
                    "support_label_count": 1,
                    "supply_label_count": 0,
                    "high_volume_up_count": 1,
                    "high_volume_down_count": 0,
                    "high_volume_stall_count": 0,
                    "long_upper_shadow_count": 0,
                    "long_lower_shadow_count": 1,
                    "low_volume_pullback_count": 0,
                    "low_volume_rebound_count": 1,
                    "breakout_like_count": 1,
                    "breakdown_like_count": 0,
                    "last_part_bull_score": 0.7,
                    "previous_part_bull_score": 0.5,
                    "bull_score_change": 0.2,
                    "sequence_pattern": "HEALTHY_UPTREND_PATTERN",
                    "sequence_strength_score": 0.8,
                }
            )
        states.append(
            {
                "date": row.trade_date,
                "scope_type": "stock",
                "scope_id": row.code,
                "state_10": "NEUTRAL",
                "state_20": "HEALTHY_UPTREND",
                "state_30": None,
                "state_60": None,
                "state_120": None,
                "state_240": None,
                "final_state": "HEALTHY_UPTREND",
                "trend_background": "UPTREND",
                "position_background": "MID",
                "market_score": 70.0,
                "sector_score": 75.0,
                "self_score": 80.0,
                "relative_strength_score": 65.0,
                "resonance_score": 60.0,
                "final_rating": "A",
                "confidence": 0.8,
                "main_features": "fixture",
                "risk_flags": "",
                "bullish_confirm_condition": "",
                "bearish_invalidate_condition": "",
            }
        )
    upsert_dataframe(con, "vpa_features", pd.DataFrame(features), ["date", "scope_type", "scope_id", "window_n"])
    upsert_dataframe(con, "vpa_bar_context_labels", pd.DataFrame(labels), ["date", "scope_type", "scope_id", "window_n"])
    upsert_dataframe(con, "vpa_sequence_stats", pd.DataFrame(sequences), ["date", "scope_type", "scope_id", "window_n"])
    upsert_dataframe(con, "vpa_structure_state", pd.DataFrame(states), ["date", "scope_type", "scope_id"])
    con.close()
    return path

