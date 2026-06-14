from __future__ import annotations


REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "risk_on": {"core": 0.55, "aggressive": 0.25, "fixed_horizon": 0.10, "cash": 0.10},
    "neutral": {"core": 0.60, "aggressive": 0.10, "fixed_horizon": 0.05, "cash": 0.25},
    "risk_off": {"core": 0.30, "aggressive": 0.0, "fixed_horizon": 0.0, "cash": 0.70},
    "crash": {"core": 0.10, "aggressive": 0.0, "fixed_horizon": 0.0, "cash": 0.90},
}


def regime_weights(final_regime: str) -> dict[str, float]:
    return dict(REGIME_WEIGHTS.get(final_regime, REGIME_WEIGHTS["neutral"]))


def drawdown_multiplier(account_drawdown: float) -> float:
    if account_drawdown <= -0.20:
        return 0.0
    if account_drawdown <= -0.15:
        return 0.25
    if account_drawdown <= -0.10:
        return 0.50
    if account_drawdown <= -0.05:
        return 0.75
    return 1.0

