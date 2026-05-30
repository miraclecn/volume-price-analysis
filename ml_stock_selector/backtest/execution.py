from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExecutionConfig:
    execution_price: str = "next_open"
    slippage_bps: float = 5.0
    commission_bps: float = 3.0
    stamp_duty_bps: float = 5.0
    a_share_lot_size: int = 100
    allow_fractional_shares: bool = True


def simulate_rebalance_orders(
    targets: pd.DataFrame,
    bars: pd.DataFrame,
    current_positions: pd.DataFrame,
    nav: float,
    config: ExecutionConfig,
) -> pd.DataFrame:
    orders = []
    bars_sorted = bars.sort_values(["code", "trade_date"])
    for target in targets.itertuples(index=False):
        future = bars_sorted[(bars_sorted["code"] == target.code) & (bars_sorted["trade_date"] > target.trade_date)]
        if future.empty:
            orders.append(_order(target, None, "buy", None, "rejected", "no_next_bar", config))
            continue
        bar = future.iloc[0]
        orders.append(_order(target, bar, "buy", nav, _status(target, bar, "buy"), _reason(bar, "buy"), config))
    out = pd.DataFrame(orders)
    assert_no_t0_fills(out)
    return out


def assert_no_t0_fills(orders: pd.DataFrame) -> None:
    bad = orders[(orders["status"] == "filled") & (orders["sim_date"] <= orders["decision_date"])]
    if not bad.empty:
        raise ValueError("T+1 execution violation: filled order has sim_date <= decision_date")


def _order(target, bar, side: str, nav: float | None, status: str, reason: str | None, config: ExecutionConfig) -> dict[str, object]:
    fill_px = None
    qty = 0.0
    sim_date = None
    if bar is not None:
        sim_date = bar["trade_date"]
        if status == "filled":
            sign = 1 if side == "buy" else -1
            fill_px = float(bar["open"]) * (1 + sign * config.slippage_bps / 10000.0)
            qty = (nav or 0.0) * float(target.target_weight) / fill_px if fill_px else 0.0
            if not config.allow_fractional_shares:
                qty = int(qty // config.a_share_lot_size) * config.a_share_lot_size
    return {
        "run_id": None,
        "sim_date": sim_date,
        "decision_date": target.trade_date,
        "code": target.code,
        "side": side,
        "qty": qty,
        "target_weight": target.target_weight,
        "order_px_ref": config.execution_price,
        "fill_px": fill_px,
        "status": status,
        "reason": reason,
    }


def _status(target, bar: pd.Series, side: str) -> str:
    return "filled" if _reason(bar, side) is None else "rejected"


def _reason(bar: pd.Series, side: str) -> str | None:
    if bool(bar.get("is_paused", False)):
        return "paused"
    if side == "buy" and float(bar["open"]) >= float(bar["limit_up"]):
        return "limit_up"
    if side == "sell" and float(bar["open"]) <= float(bar["limit_down"]):
        return "limit_down"
    return None

