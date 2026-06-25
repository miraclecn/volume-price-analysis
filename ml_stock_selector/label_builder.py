from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

import pandas as pd

from ml_stock_selector.constants import LABEL_BASE_FROM_CLOSE, LABEL_BASE_FROM_NEXT_OPEN, UNKNOWN_INDUSTRY_CODE
from ml_stock_selector.limit_bands import add_limit_band_columns


DEFAULT_FUTURE_SCORE_WEIGHTS = {
    "future_ret": 1.0,
    "future_max_gain": 0.5,
    "future_max_drawdown_abs": -0.7,
}
DEFAULT_RANK_LABEL_THRESHOLDS = [
    {"label": 4, "min_rank_pct": 0.99},
    {"label": 3, "min_rank_pct": 0.95},
    {"label": 2, "min_rank_pct": 0.90},
    {"label": 1, "min_rank_pct": 0.70},
]


def build_labels(
    normalized_bars: pd.DataFrame,
    horizons: list[int],
    risk_drawdown_threshold: float = -0.05,
    label_bases: list[str] | None = None,
    include_v2: bool = False,
    min_industry_peer_count: int = 1,
    future_score_weights: Mapping[str, object] | None = None,
    rank_label_thresholds: list[Mapping[str, object]] | None = None,
    rank_group_by_limit_band: bool = False,
) -> pd.DataFrame:
    normalized_bars = add_limit_band_columns(normalized_bars)
    score_weights = normalize_future_score_weights(future_score_weights)
    thresholds = normalize_rank_label_thresholds(rank_label_thresholds)
    label_bases = label_bases or [LABEL_BASE_FROM_CLOSE, LABEL_BASE_FROM_NEXT_OPEN]
    generated_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for code, group in normalized_bars.sort_values(["code", "trade_date"]).groupby("code", sort=False):
        prices = group.reset_index(drop=True)
        for idx, item in prices.iterrows():
            for horizon in horizons:
                future = prices.iloc[idx + 1 : idx + horizon + 1]
                if len(future) < horizon:
                    continue
                for label_base in label_bases:
                    if label_base == LABEL_BASE_FROM_CLOSE:
                        base_price = float(item["close"])
                    elif label_base == LABEL_BASE_FROM_NEXT_OPEN:
                        base_price = float(future.iloc[0]["open"])
                    else:
                        raise ValueError(f"Unknown label_base: {label_base}")
                    future_ret = _ret(base_price, float(future.iloc[-1]["close"]))
                    max_gain = _ret(base_price, float(future["high"].max()))
                    max_drawdown = _ret(base_price, float(future["low"].min()))
                    score = (
                        score_weights["future_ret"] * future_ret
                        + score_weights["future_max_gain"] * max_gain
                        + score_weights["future_max_drawdown_abs"] * abs(max_drawdown)
                    )
                    rows.append(
                        {
                            "trade_date": item["trade_date"],
                            "code": code,
                            "industry_code": item.get("industry_code"),
                            "horizon_d": horizon,
                            "label_base": label_base,
                            "limit_up_pct": item.get("limit_up_pct"),
                            "limit_down_pct": item.get("limit_down_pct"),
                            "limit_band": item.get("limit_band"),
                            "base_price": base_price,
                            "future_ret": future_ret,
                            "future_max_gain": max_gain,
                            "future_max_drawdown": max_drawdown,
                            "future_score": score,
                            "future_rank_pct": None,
                            "rank_label": None,
                            "risk_label": int(max_drawdown <= risk_drawdown_threshold),
                            "outperform_market": None,
                            "generated_at": generated_at,
                        }
                    )
    labels = pd.DataFrame(rows)
    if labels.empty:
        return labels
    rank_group_keys = _rank_group_keys(labels, rank_group_by_limit_band)
    labels["future_rank_pct"] = labels.groupby(rank_group_keys)["future_score"].rank(pct=True)
    labels["rank_label"] = labels["future_rank_pct"].map(lambda value: rank_label_from_pct(value, thresholds)).astype("int64")
    mean_ret = labels.groupby(["trade_date", "horizon_d", "label_base"])["future_ret"].transform("mean")
    labels["outperform_market"] = labels["future_ret"] > mean_ret
    if include_v2:
        labels = _add_v2_labels(labels, min_industry_peer_count, thresholds, rank_group_by_limit_band)
    return labels.sort_values(["trade_date", "code", "horizon_d", "label_base"]).reset_index(drop=True)


def rank_label_from_pct(rank_pct: float, thresholds: list[Mapping[str, object]] | None = None) -> int:
    for threshold in normalize_rank_label_thresholds(thresholds):
        if rank_pct >= float(threshold["min_rank_pct"]):
            return int(threshold["label"])
    return 0


def normalize_future_score_weights(weights: Mapping[str, object] | None = None) -> dict[str, float]:
    raw = {**DEFAULT_FUTURE_SCORE_WEIGHTS, **dict(weights or {})}
    return {
        "future_ret": float(raw["future_ret"]),
        "future_max_gain": float(raw["future_max_gain"]),
        "future_max_drawdown_abs": float(raw["future_max_drawdown_abs"]),
    }


def normalize_rank_label_thresholds(
    thresholds: list[Mapping[str, object]] | None = None,
) -> list[dict[str, float | int]]:
    raw = thresholds or DEFAULT_RANK_LABEL_THRESHOLDS
    normalized = [
        {"label": int(item["label"]), "min_rank_pct": float(item["min_rank_pct"])}
        for item in raw
    ]
    return sorted(normalized, key=lambda item: float(item["min_rank_pct"]), reverse=True)


def _ret(current: float, future: float) -> float:
    return round(float(future) / float(current) - 1.0, 12)


def _add_v2_labels(
    labels: pd.DataFrame,
    min_industry_peer_count: int,
    rank_label_thresholds: list[Mapping[str, object]],
    rank_group_by_limit_band: bool,
) -> pd.DataFrame:
    out = labels.copy()
    group_keys = ["trade_date", "horizon_d", "label_base"]
    out["absolute_ret"] = out["future_ret"]
    out["absolute_rank_pct"] = out["future_rank_pct"]
    out["absolute_label"] = out["rank_label"]
    out["market_ret"] = out.groupby(group_keys)["absolute_ret"].transform("mean")
    out["benchmark_missing_market"] = out["market_ret"].isna()
    out["market_excess_ret"] = out["absolute_ret"] - out["market_ret"]
    unknown_industry = out["industry_code"].map(_is_unknown_industry)
    industry_keys = group_keys + ["industry_code"]
    industry_sum = out.groupby(industry_keys)["absolute_ret"].transform("sum")
    industry_count = out.groupby(industry_keys)["absolute_ret"].transform("count")
    peer_count = (industry_count - 1).clip(lower=0).astype("int64")
    peer_count = peer_count.where(~unknown_industry, 0).astype("int64")
    out["benchmark_peer_count"] = peer_count
    enough_peers = out["benchmark_peer_count"] >= int(min_industry_peer_count)
    denominator = (industry_count - 1).where((industry_count - 1) > 0)
    industry_ret = (industry_sum - out["absolute_ret"]) / denominator
    out["industry_ret"] = industry_ret.where(enough_peers & ~unknown_industry)
    out["benchmark_missing_industry"] = (unknown_industry | ~enough_peers).astype(object)
    out["industry_excess_ret"] = out["absolute_ret"] - out["industry_ret"]
    out["active_score"] = out["market_excess_ret"]
    has_industry = ~out["benchmark_missing_industry"].astype(bool)
    if has_industry.any():
        out.loc[has_industry, "active_score"] = (
            0.5 * out.loc[has_industry, "market_excess_ret"]
            + 0.5 * out.loc[has_industry, "industry_excess_ret"]
        )
    out["active_rank_pct"] = out.groupby(_rank_group_keys(out, rank_group_by_limit_band))["active_score"].rank(pct=True)
    out["active_label"] = out["active_rank_pct"].map(lambda value: rank_label_from_pct(value, rank_label_thresholds)).astype("int64")
    out["benchmark_missing_market"] = out["benchmark_missing_market"].astype(object)
    return out


def _rank_group_keys(labels: pd.DataFrame, rank_group_by_limit_band: bool) -> list[str]:
    keys = ["trade_date", "horizon_d", "label_base"]
    if rank_group_by_limit_band and "limit_band" in labels.columns:
        keys.append("limit_band")
    return keys


def _is_unknown_industry(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip().upper() == UNKNOWN_INDUSTRY_CODE
