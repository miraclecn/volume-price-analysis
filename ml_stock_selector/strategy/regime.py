from __future__ import annotations


def classify_market_regime(
    *,
    trend_score: float,
    breadth_score: float,
    sentiment_score: float,
    liquidity_score: float,
    volatility_score: float,
) -> str:
    risk_appetite = (float(trend_score) + float(breadth_score) + float(sentiment_score) + float(liquidity_score)) / 4.0
    volatility = float(volatility_score)
    if risk_appetite <= 0.25 or volatility >= 0.85:
        return "crash"
    if risk_appetite <= 0.40 or volatility >= 0.70:
        return "risk_off"
    if risk_appetite >= 0.65 and volatility <= 0.50:
        return "risk_on"
    return "neutral"

