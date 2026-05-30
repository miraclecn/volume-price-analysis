from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ml_stock_selector.constants import LABEL_BASE_FROM_CLOSE, LABEL_BASE_FROM_NEXT_OPEN


def build_labels(
    normalized_bars: pd.DataFrame,
    horizons: list[int],
    risk_drawdown_threshold: float = -0.05,
    label_bases: list[str] | None = None,
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

