from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml_stock_selector.backtest.execution import ExecutionConfig, simulate_rebalance_orders
from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constraints import PortfolioConstraints
from ml_stock_selector.portfolio.constructor import construct_portfolio_targets_v2
from ml_stock_selector.portfolio.constructor import get_portfolio_diagnostics


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float
    portfolio_id: str
    execution: ExecutionConfig
    decision_dates: list[str] | None = None


@dataclass(frozen=True)
class BacktestResult:
    orders: pd.DataFrame
    positions: pd.DataFrame
    nav: pd.DataFrame
    portfolio_diagnostics: pd.DataFrame | None = None


def run_holding_aware_backtest(
    scored_candidates: pd.DataFrame,
    bars: pd.DataFrame,
    constraints: PortfolioConstraints,
    config: BacktestConfig,
    *,
    min_weight: float = 0.05,
    max_weight: float = 0.10,
    allow_cash: bool = True,
    run_id: str | None = None,
    fold_id: str | None = None,
    score_version: str | None = "v2_three_model",
) -> BacktestResult:
    built_targets: list[pd.DataFrame] = []
    built_diagnostics: list[pd.DataFrame] = []

    def build_targets(date: str, positions: dict[str, float], position_meta: dict[str, dict[str, object]]) -> pd.DataFrame:
        day_candidates = scored_candidates[scored_candidates["trade_date"] == date].copy()
        if day_candidates.empty:
            return pd.DataFrame(columns=["trade_date", "portfolio_id", "code", "target_weight"])
        targets = construct_portfolio_targets_v2(
            day_candidates,
            constraints,
            config.portfolio_id,
            current_holdings=_holdings_frame(positions, position_meta, date),
            run_id=run_id,
            fold_id=fold_id,
            score_version=score_version,
        )
        weighted = allocate_weights(targets, min_weight, max_weight, allow_cash)
        built_targets.append(weighted)
        built_diagnostics.append(get_portfolio_diagnostics(weighted))
        return weighted

    result = _run_backtest_loop(
        pd.DataFrame(columns=["trade_date", "portfolio_id", "code", "target_weight"]),
        bars,
        config,
        target_builder=build_targets,
        decision_dates=set(scored_candidates["trade_date"].dropna().unique()) if "trade_date" in scored_candidates else set(),
    )
    targets = pd.concat([_without_attrs(frame) for frame in built_targets], ignore_index=True) if built_targets else pd.DataFrame()
    diagnostics = pd.concat(built_diagnostics, ignore_index=True) if built_diagnostics else pd.DataFrame()
    return BacktestResult(result.orders, result.positions, result.nav, diagnostics if not diagnostics.empty else get_portfolio_diagnostics(targets))


def run_backtest(targets: pd.DataFrame, bars: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    return _run_backtest_loop(targets, bars, config)


def _run_backtest_loop(
    targets: pd.DataFrame,
    bars: pd.DataFrame,
    config: BacktestConfig,
    *,
    target_builder=None,
    decision_dates: set[str] | None = None,
) -> BacktestResult:
    cash = config.initial_cash
    positions: dict[str, float] = {}
    position_meta: dict[str, dict[str, object]] = {}
    all_orders: list[pd.DataFrame] = []
    nav_rows = []
    position_rows = []
    if bars.empty:
        return BacktestResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), get_portfolio_diagnostics(targets))
    bars_sorted = bars.sort_values(["code", "trade_date"]).copy()
    bars_sorted.attrs["_vpa_sorted_by_code_date"] = True
    trading_dates = sorted(bars_sorted["trade_date"].dropna().unique())
    targets_by_date = {
        date: day_targets
        for date, day_targets in targets.sort_values("trade_date").groupby("trade_date", sort=True)
    } if not targets.empty else {}
    decision_dates = decision_dates if decision_dates is not None else (set(config.decision_dates) if config.decision_dates is not None else set(targets_by_date))
    pending_orders: list[dict[str, object]] = []
    for date in trading_dates:
        todays_orders = [order for order in pending_orders if order.get("sim_date") == date]
        pending_orders = [order for order in pending_orders if order.get("sim_date") != date]
        turnover_value = 0.0
        for order in todays_orders:
            if order.get("status") != "filled":
                continue
            qty = float(order["qty"])
            fill_px = float(order["fill_px"])
            value = qty * fill_px
            turnover_value += value
            code = str(order["code"])
            if order.get("side") == "sell":
                cash += value
                next_qty = positions.get(code, 0.0) - qty
            else:
                cash -= value
                next_qty = positions.get(code, 0.0) + qty
                position_meta.setdefault(
                    code,
                    {
                        "entry_date": order.get("sim_date"),
                        "entry_price": fill_px,
                        "entry_trade_score": order.get("entry_trade_score"),
                        "entry_reason": order.get("entry_reason"),
                    },
                )
            if abs(next_qty) <= 1e-9:
                positions.pop(code, None)
                position_meta.pop(code, None)
            else:
                positions[code] = next_qty
        gross = _position_value(positions, bars_sorted, date)
        nav = cash + gross
        nav_rows.append(
            {
                "run_id": None,
                "sim_date": date,
                "nav": nav,
                "cash": cash,
                "gross_exposure": gross / nav if nav else 0.0,
                "turnover": turnover_value / nav if nav else 0.0,
            }
        )
        for code, qty in positions.items():
            px = _mark_price(bars_sorted, code, date)
            meta = position_meta.get(code, {})
            position_rows.append(
                {
                    "run_id": None,
                    "sim_date": date,
                    "code": code,
                    "position_qty": qty,
                    "market_value": qty * px,
                    "weight": (qty * px / nav) if nav else 0.0,
                    "entry_date": meta.get("entry_date"),
                    "entry_price": meta.get("entry_price"),
                    "holding_days": _holding_days(meta.get("entry_date"), date),
                    "entry_trade_score": meta.get("entry_trade_score"),
                    "entry_reason": meta.get("entry_reason"),
                }
            )
        if date in decision_dates:
            pending_codes = {str(order["code"]) for order in pending_orders}
            day_targets = target_builder(date, positions, position_meta) if target_builder is not None else targets_by_date.get(date, pd.DataFrame(columns=targets.columns))
            orders = simulate_rebalance_orders(
                day_targets,
                bars_sorted,
                _positions_frame(positions, position_meta),
                nav,
                config.execution,
                decision_date=date,
            )
            if pending_codes and not orders.empty:
                orders = orders[~orders["code"].astype(str).isin(pending_codes)].reset_index(drop=True)
            all_orders.append(orders)
            if not orders.empty:
                pending_orders.extend(orders.to_dict("records"))
    return BacktestResult(
        pd.concat(all_orders, ignore_index=True) if all_orders else pd.DataFrame(),
        pd.DataFrame(position_rows),
        pd.DataFrame(nav_rows),
        get_portfolio_diagnostics(targets),
    )


def _positions_frame(positions: dict[str, float], position_meta: dict[str, dict[str, object]] | None = None) -> pd.DataFrame:
    if not positions:
        return pd.DataFrame(columns=["code", "position_qty", "entry_date", "entry_price", "entry_trade_score", "entry_reason"])
    position_meta = position_meta or {}
    return pd.DataFrame(
        [
            {
                "code": code,
                "position_qty": qty,
                **position_meta.get(code, {}),
            }
            for code, qty in positions.items()
        ]
    )


def _holdings_frame(positions: dict[str, float], position_meta: dict[str, dict[str, object]], date: str) -> pd.DataFrame:
    frame = _positions_frame(positions, position_meta)
    if frame.empty:
        return frame
    frame = frame.rename(columns={"position_qty": "shares"}).copy()
    frame["holding_days"] = [
        _holding_days(row.get("entry_date"), date) or 0
        for _, row in frame.iterrows()
    ]
    frame["calendar_days"] = frame["holding_days"]
    frame["latest_trade_score"] = frame.get("entry_trade_score")
    return frame


def _without_attrs(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.attrs.clear()
    return out


def _position_value(positions: dict[str, float], bars: pd.DataFrame, date: str) -> float:
    return sum(qty * _mark_price(bars, code, date) for code, qty in positions.items())


def _mark_price(bars: pd.DataFrame, code: str, date: str) -> float:
    subset = bars[(bars["code"] == code) & (bars["trade_date"] <= date)].sort_values("trade_date")
    if subset.empty:
        return 0.0
    return float(subset.iloc[-1]["close"])


def _holding_days(entry_date: object, current_date: object) -> int | None:
    start = pd.to_datetime(entry_date, errors="coerce")
    end = pd.to_datetime(current_date, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return None
    return max(int((end - start).days), 0)
