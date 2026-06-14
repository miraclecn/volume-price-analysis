from __future__ import annotations

import pandas as pd

from ml_stock_selector.strategy.ensemble import StrategySleeve
from ml_stock_selector.strategy.risk_budget import drawdown_multiplier, regime_weights


def allocate_strategy_sleeves(
    *,
    trade_date: str,
    sleeves: list[StrategySleeve],
    final_regime: str,
    account_drawdown: float,
    health_enabled_by_bundle: dict[str, bool] | None = None,
    generated_at: str | None = None,
) -> pd.DataFrame:
    health_enabled_by_bundle = health_enabled_by_bundle or {}
    weights = regime_weights(final_regime)
    drawdown_scale = drawdown_multiplier(account_drawdown)
    rows: list[dict[str, object]] = []
    non_cash_total = 0.0
    cash_sleeves = [sleeve for sleeve in sleeves if sleeve.sleeve == "cash"]

    for sleeve in sleeves:
        if sleeve.sleeve == "cash":
            continue
        raw_weight = weights.get(sleeve.sleeve, 0.0)
        health_multiplier = _health_multiplier(sleeve, health_enabled_by_bundle)
        enabled_multiplier = 1.0 if sleeve.enabled else 0.0
        final_weight = _clean_weight(raw_weight * health_multiplier * enabled_multiplier * drawdown_scale)
        non_cash_total += final_weight
        rows.append(
            _allocation_row(
                trade_date=trade_date,
                sleeve=sleeve,
                raw_weight=raw_weight,
                regime_multiplier=1.0 if raw_weight > 0.0 else 0.0,
                health_multiplier=health_multiplier,
                drawdown_multiplier=drawdown_scale,
                final_weight=final_weight,
                reason=_reason(final_regime, sleeve, health_multiplier, drawdown_scale),
                generated_at=generated_at,
            )
        )

    cash_weight = _clean_weight(max(0.0, 1.0 - non_cash_total))
    cash = cash_sleeves[0] if cash_sleeves else StrategySleeve("cash", "cash_reserve", "cash", None)
    rows.append(
        _allocation_row(
            trade_date=trade_date,
            sleeve=cash,
            raw_weight=weights.get("cash", 0.0),
            regime_multiplier=1.0,
            health_multiplier=1.0,
            drawdown_multiplier=1.0,
            final_weight=cash_weight,
            reason=f"{final_regime}: residual risk budget held as cash",
            generated_at=generated_at,
        )
    )
    return pd.DataFrame(rows)


def _health_multiplier(sleeve: StrategySleeve, health_enabled_by_bundle: dict[str, bool]) -> float:
    if sleeve.bundle_id is None:
        return 1.0
    return 1.0 if health_enabled_by_bundle.get(sleeve.bundle_id, True) else 0.0


def _allocation_row(
    *,
    trade_date: str,
    sleeve: StrategySleeve,
    raw_weight: float,
    regime_multiplier: float,
    health_multiplier: float,
    drawdown_multiplier: float,
    final_weight: float,
    reason: str,
    generated_at: str | None,
) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "strategy_id": sleeve.strategy_id,
        "sleeve": sleeve.sleeve,
        "bundle_id": sleeve.bundle_id or sleeve.strategy_id,
        "score_version": sleeve.score_version,
        "raw_weight": _clean_weight(raw_weight),
        "regime_multiplier": _clean_weight(regime_multiplier),
        "health_multiplier": _clean_weight(health_multiplier),
        "drawdown_multiplier": _clean_weight(drawdown_multiplier),
        "final_weight": _clean_weight(final_weight),
        "reason": reason,
        "generated_at": generated_at,
    }


def _reason(final_regime: str, sleeve: StrategySleeve, health_multiplier: float, drawdown_scale: float) -> str:
    parts = [final_regime]
    if health_multiplier == 0.0:
        parts.append("disabled_by_health")
    if drawdown_scale < 1.0:
        parts.append("drawdown_scaled")
    if not sleeve.enabled:
        parts.append("disabled_by_config")
    return "; ".join(parts)


def _clean_weight(value: float) -> float:
    return round(float(value), 12)
