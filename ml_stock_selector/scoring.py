from __future__ import annotations

import pandas as pd

from ml_stock_selector.models.calibrator import cross_sectional_percentile


def add_context_score(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    sources = [col for col in ["confidence", "self_score", "sector_score", "market_score", "resonance_score"] if col in out]
    if sources:
        out["context_score"] = out[sources].mean(axis=1)
        out["context_score_pct"] = cross_sectional_percentile(out, "context_score")
    else:
        out["context_score"] = 0.5
        out["context_score_pct"] = 0.5
    if "resonance_pct" not in out:
        out["resonance_pct"] = cross_sectional_percentile(out, "resonance_score") if "resonance_score" in out else 0.5
    return out


def add_liquidity_score(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    sources = [col for col in ["adv20_amount", "amount", "turnover_rate"] if col in out]
    if sources:
        out["liquidity_score"] = out[sources].mean(axis=1)
        out["liquidity_score_pct"] = cross_sectional_percentile(out, "liquidity_score")
    else:
        out["liquidity_score"] = 0.5
        out["liquidity_score_pct"] = 0.5
    return out


def score_candidates(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    if "alpha_rank_pct" not in out or out["alpha_rank_pct"].isna().any():
        out["alpha_rank_pct"] = cross_sectional_percentile(out, "alpha_score")
    if "risk_score" not in out:
        out["risk_score"] = 0.0
    if "risk_rank_pct" not in out or out["risk_rank_pct"].isna().any():
        out["risk_rank_pct"] = cross_sectional_percentile(out, "risk_score")
    if "context_score_pct" not in out:
        out = add_context_score(out)
    if "liquidity_score_pct" not in out:
        out = add_liquidity_score(out)
    for column, default in {"relative_strength_pct": 0.5, "resonance_pct": 0.5, "penalty_score": 0.0}.items():
        if column not in out:
            out[column] = default
    out["trade_score"] = (
        0.60 * out["alpha_rank_pct"]
        + 0.15 * out["context_score_pct"]
        + 0.10 * out["liquidity_score_pct"]
        + 0.05 * out["relative_strength_pct"]
        + 0.10 * out["resonance_pct"]
        - 0.30 * out["risk_rank_pct"]
        - out["penalty_score"]
    )
    return out.sort_values(["trade_date", "trade_score", "code"], ascending=[True, False, True]).reset_index(drop=True)

