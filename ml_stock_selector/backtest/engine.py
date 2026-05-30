from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml_stock_selector.backtest.execution import ExecutionConfig, simulate_rebalance_orders


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float
    portfolio_id: str
    execution: ExecutionConfig


@dataclass(frozen=True)
class BacktestResult:
    orders: pd.DataFrame
    positions: pd.DataFrame
    nav: pd.DataFrame


def run_backtest(targets: pd.DataFrame, bars: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    cash = config.initial_cash
    positions: dict[str, float] = {}
    all_orders = []
    nav_rows = []
    position_rows = []
    for date, day_targets in targets.sort_values("trade_date").groupby("trade_date", sort=True):
        orders = simulate_rebalance_orders(day_targets, bars, pd.DataFrame(), cash + _position_value(positions, bars, date), config.execution)
        all_orders.append(orders)
        for order in orders.itertuples(index=False):
            if order.status == "filled":
                cost = float(order.qty) * float(order.fill_px)
                cash -= cost
                positions[order.code] = positions.get(order.code, 0.0) + float(order.qty)
        mark_date = orders["sim_date"].dropna().max() if not orders.empty else date
        gross = _position_value(positions, bars, mark_date)
        nav = cash + gross
        nav_rows.append({"run_id": None, "sim_date": mark_date, "nav": nav, "cash": cash, "gross_exposure": gross / nav if nav else 0.0, "turnover": 0.0})
        for code, qty in positions.items():
            px = _mark_price(bars, code, mark_date)
            position_rows.append({"run_id": None, "sim_date": mark_date, "code": code, "position_qty": qty, "market_value": qty * px, "weight": (qty * px / nav) if nav else 0.0})
    return BacktestResult(
        pd.concat(all_orders, ignore_index=True) if all_orders else pd.DataFrame(),
        pd.DataFrame(position_rows),
        pd.DataFrame(nav_rows),
    )


def _position_value(positions: dict[str, float], bars: pd.DataFrame, date: str) -> float:
    return sum(qty * _mark_price(bars, code, date) for code, qty in positions.items())


def _mark_price(bars: pd.DataFrame, code: str, date: str) -> float:
    subset = bars[(bars["code"] == code) & (bars["trade_date"] <= date)].sort_values("trade_date")
    if subset.empty:
        return 0.0
    return float(subset.iloc[-1]["close"])

