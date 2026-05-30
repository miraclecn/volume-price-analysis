from __future__ import annotations

import math

import pandas as pd


def max_drawdown(nav: pd.DataFrame, nav_col: str = "nav") -> float:
    values = nav[nav_col].astype(float)
    drawdown = values / values.cummax() - 1.0
    return float(drawdown.min())


def annualized_return(nav: pd.DataFrame, nav_col: str = "nav", periods_per_year: int = 252) -> float:
    values = nav[nav_col].astype(float)
    if len(values) < 2:
        return 0.0
    returns = values.pct_change().dropna()
    if returns.empty:
        return 0.0
    return float((1.0 + returns.mean()) ** periods_per_year - 1.0)


def rank_ic(frame: pd.DataFrame, pred_col: str, label_col: str) -> float:
    return float(frame[pred_col].rank().corr(frame[label_col].rank()))


def ndcg_at_k(frame: pd.DataFrame, pred_col: str, label_col: str, k: int) -> float:
    ordered = frame.sort_values(pred_col, ascending=False).head(k)
    ideal = frame.sort_values(label_col, ascending=False).head(k)
    dcg = _dcg(ordered[label_col].tolist())
    idcg = _dcg(ideal[label_col].tolist())
    return float(dcg / idcg) if idcg else 0.0


def _dcg(values: list[float]) -> float:
    return sum((2**float(value) - 1.0) / math.log2(idx + 2) for idx, value in enumerate(values))
