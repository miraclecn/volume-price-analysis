from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_SHEETS = [
    "全A市场结构",
    "板块结构排名",
    "强势板块候选",
    "个股结构总表",
    "三层共振个股",
    "低位承接增强个股",
    "健康上涨个股",
    "高位供应风险个股",
    "放量破位风险个股",
    "个股多窗口标签明细",
    "后效验证数据",
]


def export_excel_report(
    output_path: Path | str,
    market: pd.DataFrame,
    sectors: pd.DataFrame,
    stocks: pd.DataFrame,
    labels: pd.DataFrame,
    validation: pd.DataFrame,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheets = {
        "全A市场结构": market,
        "板块结构排名": sectors,
        "强势板块候选": _filter_rating(sectors, {"A", "B"}),
        "个股结构总表": stocks,
        "三层共振个股": _filter_rating(stocks, {"A"}),
        "低位承接增强个股": _filter_state(
            stocks, {"LOW_LEVEL_SUPPORT", "POSSIBLE_ACCUMULATION"}
        ),
        "健康上涨个股": _filter_state(stocks, {"HEALTHY_UPTREND"}),
        "高位供应风险个股": _filter_state(
            stocks, {"HIGH_LEVEL_SUPPLY", "POSSIBLE_DISTRIBUTION"}
        ),
        "放量破位风险个股": _filter_breakdown(stocks),
        "个股多窗口标签明细": _with_leading_columns(
            labels, ["date", "scope_type", "scope_id", "window_n"]
        ),
        "后效验证数据": validation,
    }
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name in REQUIRED_SHEETS:
            sheets[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
    return path


def _filter_rating(frame: pd.DataFrame, ratings: set[str]) -> pd.DataFrame:
    if "final_rating" not in frame.columns:
        return frame.iloc[0:0]
    return frame[frame["final_rating"].isin(ratings)]


def _filter_state(frame: pd.DataFrame, states: set[str]) -> pd.DataFrame:
    if "final_state" not in frame.columns:
        return frame.iloc[0:0]
    return frame[frame["final_state"].isin(states)]


def _filter_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    if "final_state" not in frame.columns:
        return frame.iloc[0:0]
    risk_flags = frame.get("risk_flags", pd.Series([""] * len(frame))).fillna("")
    return frame[(frame["final_state"] == "BREAKDOWN") | risk_flags.str.contains("破位")]


def _with_leading_columns(frame: pd.DataFrame, leading_columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in leading_columns:
        if column not in output.columns:
            output[column] = None
    remaining = [column for column in output.columns if column not in leading_columns]
    return output[leading_columns + remaining]
