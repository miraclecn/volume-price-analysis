from __future__ import annotations

from datetime import datetime, timezone
import json

import duckdb
import pandas as pd

from ml_stock_selector.constants import (
    FEATURE_SET_BASELINE_A,
    FEATURE_SET_BASELINE_B,
    FEATURE_SET_VPA_C,
    FEATURE_SET_VPA_D,
    FEATURE_SET_VPA_E,
    UNKNOWN_INDUSTRY_CODE,
)
from ml_stock_selector.ohlcv_features import build_ohlcv_features


OHLCV_FEATURE_PREFIXES = (
    "ret_",
    "open_gap_pct",
    "range_pct",
    "body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "close_position",
    "amount",
    "turnover_rate",
    "volatility_",
    "amount_ratio_",
    "volume_ratio_",
    "turnover_mean_",
    "high_distance_",
    "low_distance_",
)
VPA_NUMERIC_COLUMNS = ["ret_pct", "range_pct", "body_pct", "vol_rvol_n", "range_rvol_n", "price_position_n", "ma_slope_n"]
BAR_CONTEXT_COLUMNS = ["raw_label", "bull_bear_score", "supply_score", "demand_score", "volatility_score"]
SEQUENCE_COLUMNS = ["abnormal_ratio", "support_label_count", "supply_label_count", "bull_score_change", "sequence_strength_score", "sequence_pattern"]
STRUCTURE_COLUMNS = ["final_state", "final_rating", "confidence", "market_score", "sector_score", "self_score", "relative_strength_score", "resonance_score"]


def build_vpa_numeric_features(vpa_db_path: str, start_date: str, end_date: str, windows: list[int]) -> pd.DataFrame:
    frame = _read_vpa(vpa_db_path, "vpa_features", start_date, end_date, windows)
    return _pivot_by_window(frame, VPA_NUMERIC_COLUMNS)


def build_vpa_bar_context_features(vpa_db_path: str, start_date: str, end_date: str, windows: list[int]) -> pd.DataFrame:
    frame = _read_vpa(vpa_db_path, "vpa_bar_context_labels", start_date, end_date, windows)
    return _pivot_by_window(frame, BAR_CONTEXT_COLUMNS)


def build_vpa_sequence_features(vpa_db_path: str, start_date: str, end_date: str, windows: list[int]) -> pd.DataFrame:
    frame = _read_vpa(vpa_db_path, "vpa_sequence_stats", start_date, end_date, windows)
    return _pivot_by_window(frame, SEQUENCE_COLUMNS)


def build_structure_state_features(vpa_db_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    con = duckdb.connect(vpa_db_path, read_only=True)
    try:
        frame = con.execute(
            """
            select *
            from vpa_structure_state
            where scope_type = 'stock' and date between ? and ?
            order by date, scope_id
            """,
            [start_date, end_date],
        ).fetchdf()
    finally:
        con.close()
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", "code"])
    columns = ["date", "scope_id"] + [col for col in STRUCTURE_COLUMNS if col in frame.columns]
    return frame[columns].rename(columns={"date": "trade_date", "scope_id": "code"})


def apply_feature_set_filter(features: pd.DataFrame, feature_set_id: str) -> pd.DataFrame:
    protected = {"trade_date", "code"}
    if feature_set_id == FEATURE_SET_VPA_E:
        return features.copy()
    excluded = set(STRUCTURE_COLUMNS)
    if feature_set_id == FEATURE_SET_VPA_D:
        return features[[col for col in features.columns if col in protected or col not in excluded]].copy()
    if feature_set_id == FEATURE_SET_VPA_C:
        return features[[col for col in features.columns if col in protected or (col not in excluded and not _is_sequence(col))]].copy()
    if feature_set_id == FEATURE_SET_BASELINE_B:
        return features[[col for col in features.columns if col in protected or (_is_ohlcv(col) or _is_vpa_numeric(col))]].copy()
    if feature_set_id == FEATURE_SET_BASELINE_A:
        return features[[col for col in features.columns if col in protected or _is_ohlcv(col)]].copy()
    raise ValueError(f"Unknown feature_set_id: {feature_set_id}")


def build_feature_mart(
    vpa_db_path: str,
    normalized_bars: pd.DataFrame,
    start_date: str,
    end_date: str,
    feature_set_id: str,
    windows: list[int],
    tradeability: pd.DataFrame,
    exclude_industry_metadata_from_features_json: bool = False,
) -> pd.DataFrame:
    feature_input = normalized_bars[normalized_bars["trade_date"] <= end_date]
    ohlcv_all = build_ohlcv_features(feature_input, windows)
    ohlcv = ohlcv_all[(ohlcv_all["trade_date"] >= start_date) & (ohlcv_all["trade_date"] <= end_date)].reset_index(drop=True)
    base_cols = ["trade_date", "code"]
    feature_cols = [
        col
        for col in ohlcv.columns
        if col not in set(feature_input.columns) or col in {"amount", "turnover_rate"}
    ]
    wide = ohlcv[base_cols + sorted(set(feature_cols) - set(base_cols))].copy()
    if feature_set_id in {FEATURE_SET_BASELINE_B, FEATURE_SET_VPA_C, FEATURE_SET_VPA_D, FEATURE_SET_VPA_E}:
        wide = _merge(wide, build_vpa_numeric_features(vpa_db_path, start_date, end_date, windows))
    if feature_set_id in {FEATURE_SET_VPA_C, FEATURE_SET_VPA_D, FEATURE_SET_VPA_E}:
        wide = _merge(wide, build_vpa_bar_context_features(vpa_db_path, start_date, end_date, windows))
    if feature_set_id in {FEATURE_SET_VPA_D, FEATURE_SET_VPA_E}:
        wide = _merge(wide, build_vpa_sequence_features(vpa_db_path, start_date, end_date, windows))
    if feature_set_id == FEATURE_SET_VPA_E:
        wide = _merge(wide, build_structure_state_features(vpa_db_path, start_date, end_date))
    wide = apply_feature_set_filter(wide, feature_set_id)
    generated_at = datetime.now(timezone.utc).isoformat()
    trade_cols = [
        "trade_date",
        "code",
        "industry_code",
        "industry_name",
        "is_st",
        "is_paused",
        "limit_up",
        "limit_down",
        "limit_up_pct",
        "limit_down_pct",
        "limit_band",
        "adv20_amount",
        "can_buy_next_open",
        "can_sell_next_open",
        "is_bse",
    ]
    out = wide[["trade_date", "code"]].merge(tradeability[trade_cols], on=["trade_date", "code"], how="left")
    json_source = wide.copy()
    if not exclude_industry_metadata_from_features_json:
        json_source = json_source.merge(
            out[["trade_date", "code", "industry_code", "industry_name"]],
            on=["trade_date", "code"],
            how="left",
        )
        json_source["industry_unknown"] = (
            json_source["industry_code"].fillna("").astype(str).str.upper() == UNKNOWN_INDUSTRY_CODE
        )
    json_cols = [col for col in json_source.columns if col not in {"trade_date", "code"}]
    out["feature_set_id"] = feature_set_id
    out["vpa_data_version"] = "v1"
    out["generated_at"] = generated_at
    out["features_json"] = json_source[json_cols].apply(_json_row, axis=1)
    columns = [
        "trade_date",
        "code",
        "feature_set_id",
        "vpa_data_version",
        "generated_at",
        "industry_code",
        "industry_name",
        "is_st",
        "is_paused",
        "limit_up",
        "limit_down",
        "limit_up_pct",
        "limit_down_pct",
        "limit_band",
        "adv20_amount",
        "can_buy_next_open",
        "can_sell_next_open",
        "is_bse",
        "features_json",
    ]
    return out[columns].sort_values(["trade_date", "code"]).reset_index(drop=True)


def _read_vpa(vpa_db_path: str, table: str, start_date: str, end_date: str, windows: list[int]) -> pd.DataFrame:
    con = duckdb.connect(vpa_db_path, read_only=True)
    try:
        frame = con.execute(
            f"""
            select *
            from {table}
            where scope_type = 'stock'
              and date between ? and ?
              and window_n in ({','.join(['?'] * len(windows))})
            order by date, scope_id, window_n
            """,
            [start_date, end_date, *windows],
        ).fetchdf()
    finally:
        con.close()
    return frame


def _pivot_by_window(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", "code"])
    available = [col for col in value_columns if col in frame.columns]
    if not available:
        return (
            frame[["date", "scope_id"]]
            .drop_duplicates()
            .rename(columns={"date": "trade_date", "scope_id": "code"})
        )
    wide = frame[["date", "scope_id", "window_n", *available]].pivot(
        index=["date", "scope_id"],
        columns="window_n",
        values=available,
    )
    wide.columns = [f"{column}_{int(window)}" for column, window in wide.columns]
    return wide.reset_index().rename(columns={"date": "trade_date", "scope_id": "code"})


def _merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if right.empty:
        return left
    return left.merge(right, on=["trade_date", "code"], how="left")


def _json_row(row: pd.Series) -> str:
    clean = {
        key: (0.0 if pd.isna(value) else value)
        for key, value in row.to_dict().items()
    }
    return json.dumps(clean, ensure_ascii=False, sort_keys=True)


def _is_ohlcv(column: str) -> bool:
    return any(column.startswith(prefix) or column == prefix for prefix in OHLCV_FEATURE_PREFIXES)


def _is_vpa_numeric(column: str) -> bool:
    return any(column.startswith(f"{name}_") for name in VPA_NUMERIC_COLUMNS)


def _is_sequence(column: str) -> bool:
    return any(column.startswith(f"{name}_") for name in SEQUENCE_COLUMNS)
