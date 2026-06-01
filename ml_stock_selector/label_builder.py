from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ml_stock_selector.constants import LABEL_BASE_FROM_CLOSE, LABEL_BASE_FROM_NEXT_OPEN, UNKNOWN_INDUSTRY_CODE


def build_labels(
    normalized_bars: pd.DataFrame,
    horizons: list[int],
    risk_drawdown_threshold: float = -0.05,
    label_bases: list[str] | None = None,
    include_v2: bool = False,
    min_industry_peer_count: int = 1,
) -> pd.DataFrame:
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
                    score = future_ret + 0.5 * max_gain - 0.7 * abs(max_drawdown)
                    rows.append(
                        {
                            "trade_date": item["trade_date"],
                            "code": code,
                            "industry_code": item.get("industry_code"),
                            "horizon_d": horizon,
                            "label_base": label_base,
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
    labels["future_rank_pct"] = labels.groupby(["trade_date", "horizon_d", "label_base"])["future_score"].rank(pct=True)
    labels["rank_label"] = labels["future_rank_pct"].map(rank_label_from_pct).astype("int64")
    mean_ret = labels.groupby(["trade_date", "horizon_d", "label_base"])["future_ret"].transform("mean")
    labels["outperform_market"] = labels["future_ret"] > mean_ret
    if include_v2:
        labels = _add_v2_labels(labels, min_industry_peer_count)
    return labels.sort_values(["trade_date", "code", "horizon_d", "label_base"]).reset_index(drop=True)


def rank_label_from_pct(rank_pct: float) -> int:
    if rank_pct >= 0.99:
        return 4
    if rank_pct >= 0.95:
        return 3
    if rank_pct >= 0.90:
        return 2
    if rank_pct >= 0.70:
        return 1
    return 0


def _ret(current: float, future: float) -> float:
    return round(float(future) / float(current) - 1.0, 12)


def _add_v2_labels(labels: pd.DataFrame, min_industry_peer_count: int) -> pd.DataFrame:
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
    out["active_rank_pct"] = out.groupby(group_keys)["active_score"].rank(pct=True)
    out["active_label"] = out["active_rank_pct"].map(rank_label_from_pct).astype("int64")
    out["benchmark_missing_market"] = out["benchmark_missing_market"].astype(object)
    return out


def _is_unknown_industry(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip().upper() == UNKNOWN_INDUSTRY_CODE
