from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml_stock_selector.backtest.execution import BarDataIndex, ExecutionConfig, simulate_rebalance_orders
from ml_stock_selector.portfolio.allocator import allocate_weights
from ml_stock_selector.portfolio.constraints import FixedHorizonRiskFilterConfig, PortfolioConstraints
from ml_stock_selector.portfolio.constructor import construct_portfolio_targets_v2
from ml_stock_selector.portfolio.constructor import get_portfolio_diagnostics
from ml_stock_selector.portfolio.fixed_horizon import construct_fixed_5d_risk_filter_targets


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
    candidates_by_date = _group_by_trade_date(scored_candidates)

    def build_targets(date: str, positions: dict[str, float], position_meta: dict[str, dict[str, object]]) -> pd.DataFrame:
        day_candidates = candidates_by_date.get(_date_key(date))
        if day_candidates is None:
            day_candidates = pd.DataFrame(columns=scored_candidates.columns)
        else:
            day_candidates = day_candidates.copy()
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
        decision_dates=_date_key_set(scored_candidates["trade_date"].dropna().unique()) if "trade_date" in scored_candidates else set(),
    )
    targets = pd.concat([_without_attrs(frame) for frame in built_targets], ignore_index=True) if built_targets else pd.DataFrame()
    diagnostics = pd.concat(built_diagnostics, ignore_index=True) if built_diagnostics else pd.DataFrame()
    return BacktestResult(result.orders, result.positions, result.nav, diagnostics if not diagnostics.empty else get_portfolio_diagnostics(targets))


def run_fixed_horizon_backtest(
    scored_candidates: pd.DataFrame,
    bars: pd.DataFrame,
    constraints: FixedHorizonRiskFilterConfig,
    config: BacktestConfig,
    *,
    run_id: str | None = None,
    fold_id: str | None = None,
) -> BacktestResult:
    built_targets: list[pd.DataFrame] = []
    built_diagnostics: list[pd.DataFrame] = []
    candidates_by_date = _group_by_trade_date(scored_candidates)

    def build_targets(date: str, positions: dict[str, float], position_meta: dict[str, dict[str, object]]) -> pd.DataFrame:
        day_candidates = candidates_by_date.get(_date_key(date))
        if day_candidates is None:
            day_candidates = pd.DataFrame(columns=scored_candidates.columns)
        else:
            day_candidates = day_candidates.copy()
        result = construct_fixed_5d_risk_filter_targets(
            day_candidates,
            _fixed_holdings_frame(positions, position_meta),
            constraints,
            date,
        )
        built_targets.append(result.targets)
        built_diagnostics.append(result.diagnostics)
        return result.targets

    result = _run_backtest_loop(
        pd.DataFrame(columns=["trade_date", "portfolio_id", "code", "target_weight"]),
        bars,
        config,
        target_builder=build_targets,
        decision_dates=_date_key_set(scored_candidates["trade_date"].dropna().unique()) if "trade_date" in scored_candidates else set(),
        use_trading_holding_days=True,
    )
    diagnostics = pd.concat(built_diagnostics, ignore_index=True) if built_diagnostics else pd.DataFrame()
    for frame in [result.orders, result.positions, result.nav]:
        if not frame.empty:
            frame["run_id"] = run_id
            frame["fold_id"] = fold_id
            frame["strategy_id"] = constraints.strategy_id
    if not result.orders.empty:
        result.orders["realized_ret"] = _realized_returns(result.orders)
        result.orders["realized_pnl"] = _realized_pnl(result.orders)
    return BacktestResult(result.orders, result.positions, result.nav, diagnostics)


def run_backtest(targets: pd.DataFrame, bars: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    return _run_backtest_loop(targets, bars, config)


def _run_backtest_loop(
    targets: pd.DataFrame,
    bars: pd.DataFrame,
    config: BacktestConfig,
    *,
    target_builder=None,
    decision_dates: set[str] | None = None,
    use_trading_holding_days: bool = False,
) -> BacktestResult:
    cash = config.initial_cash
    positions: dict[str, float] = {}
    position_meta: dict[str, dict[str, object]] = {}
    all_orders: list[dict[str, object]] = []
    nav_rows = []
    position_rows = []
    if bars.empty:
        return BacktestResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), get_portfolio_diagnostics(targets))
    bars_sorted = bars.sort_values(["code", "trade_date"]).copy()
    bars_sorted.attrs["_vpa_sorted_by_code_date"] = True
    bar_index = BarDataIndex.from_bars(bars_sorted)
    trading_dates = sorted(bars_sorted["trade_date"].dropna().unique())
    targets_by_date = {
        _date_key(date): day_targets
        for date, day_targets in targets.sort_values("trade_date").groupby("trade_date", sort=True)
    } if not targets.empty else {}
    decision_dates = decision_dates if decision_dates is not None else (_date_key_set(config.decision_dates) if config.decision_dates is not None else set(targets_by_date))
    pending_orders: list[dict[str, object]] = []
    for date in trading_dates:
        todays_orders = [order for order in pending_orders if order.get("sim_date") == date]
        pending_orders = [order for order in pending_orders if order.get("sim_date") != date]
        turnover_value = 0.0
        for order in sorted(todays_orders, key=_execution_priority):
            if order.get("status") != "filled":
                continue
            qty = float(order["qty"])
            fill_px = float(order["fill_px"])
            if order.get("side") == "buy":
                qty = _cash_limited_buy_qty(qty, fill_px, cash, config.execution)
                if qty <= 1e-12:
                    order["qty"] = 0.0
                    order["fill_px"] = None
                    order["status"] = "rejected"
                    order["reason"] = "insufficient_cash"
                    continue
                order["qty"] = qty
            value = qty * fill_px
            turnover_value += value
            code = str(order["code"])
            if order.get("side") == "sell":
                cash += value
                next_qty = positions.get(code, 0.0) - qty
            else:
                cash -= value
                next_qty = positions.get(code, 0.0) + qty
                new_meta = {
                    "entry_date": order.get("sim_date"),
                    "entry_price": fill_px,
                    "entry_trade_score": order.get("entry_trade_score"),
                    "entry_abs_rank_pct": order.get("entry_abs_rank_pct"),
                    "entry_risk_rank_pct": order.get("entry_risk_rank_pct"),
                    "entry_reason": order.get("entry_reason"),
                }
                if use_trading_holding_days:
                    new_meta["holding_days"] = 0
                    new_meta["last_holding_update"] = order.get("sim_date")
                position_meta.setdefault(code, new_meta)
            if abs(next_qty) <= 1e-9:
                positions.pop(code, None)
                position_meta.pop(code, None)
            else:
                positions[code] = next_qty
        if use_trading_holding_days:
            _advance_trading_holding_days(position_meta, date)
        gross = _position_value(positions, bar_index, date)
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
            px = _mark_price(bar_index, code, date)
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
                    "holding_days": meta.get("holding_days") if use_trading_holding_days else _holding_days(meta.get("entry_date"), date),
                    "entry_trade_score": meta.get("entry_trade_score"),
                    "entry_abs_rank_pct": meta.get("entry_abs_rank_pct"),
                    "entry_risk_rank_pct": meta.get("entry_risk_rank_pct"),
                    "entry_reason": meta.get("entry_reason"),
                }
            )
        date_key = _date_key(date)
        if date_key in decision_dates:
            pending_codes = {str(order["code"]) for order in pending_orders}
            day_targets = target_builder(date, positions, position_meta) if target_builder is not None else targets_by_date.get(date_key, pd.DataFrame(columns=targets.columns))
            orders = simulate_rebalance_orders(
                day_targets,
                bars_sorted,
                _positions_frame(positions, position_meta),
                nav,
                config.execution,
                decision_date=date,
                bar_index=bar_index,
            )
            if pending_codes and not orders.empty:
                orders = orders[~orders["code"].astype(str).isin(pending_codes)].reset_index(drop=True)
            if not orders.empty:
                records = orders.to_dict("records")
                all_orders.extend(records)
                pending_orders.extend(records)
    return BacktestResult(
        pd.DataFrame(all_orders),
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


def _fixed_holdings_frame(positions: dict[str, float], position_meta: dict[str, dict[str, object]]) -> pd.DataFrame:
    frame = _positions_frame(positions, position_meta)
    if frame.empty:
        return frame
    frame = frame.rename(columns={"position_qty": "shares"}).copy()
    if "holding_days" not in frame:
        frame["holding_days"] = 0
    frame["calendar_days"] = frame["holding_days"]
    frame["latest_trade_score"] = frame.get("entry_trade_score")
    return frame


def _advance_trading_holding_days(position_meta: dict[str, dict[str, object]], date: str) -> None:
    for meta in position_meta.values():
        entry_date = meta.get("entry_date")
        if entry_date is None or str(entry_date) >= str(date):
            continue
        if meta.get("last_holding_update") == date:
            continue
        meta["holding_days"] = int(meta.get("holding_days", 0) or 0) + 1
        meta["last_holding_update"] = date


def _realized_returns(orders: pd.DataFrame) -> pd.Series:
    values = []
    for row in orders.itertuples(index=False):
        if getattr(row, "side", None) != "sell" or getattr(row, "status", None) != "filled":
            values.append(None)
            continue
        entry_price = getattr(row, "entry_price", None)
        fill_px = getattr(row, "fill_px", None)
        try:
            values.append(float(fill_px) / float(entry_price) - 1.0 if entry_price else None)
        except (TypeError, ValueError):
            values.append(None)
    return pd.Series(values, index=orders.index)


def _realized_pnl(orders: pd.DataFrame) -> pd.Series:
    values = []
    for row in orders.itertuples(index=False):
        if getattr(row, "side", None) != "sell" or getattr(row, "status", None) != "filled":
            values.append(None)
            continue
        entry_price = getattr(row, "entry_price", None)
        fill_px = getattr(row, "fill_px", None)
        qty = getattr(row, "qty", None)
        try:
            values.append((float(fill_px) - float(entry_price)) * float(qty) if entry_price is not None else None)
        except (TypeError, ValueError):
            values.append(None)
    return pd.Series(values, index=orders.index)


def _without_attrs(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.attrs.clear()
    return out


def _group_by_trade_date(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if frame.empty or "trade_date" not in frame:
        return {}
    return {
        _date_key(date): group
        for date, group in frame.groupby("trade_date", sort=False)
    }


def _date_key_set(values) -> set[str]:
    return {_date_key(value) for value in values}


def _date_key(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def _position_value(positions: dict[str, float], bar_index: BarDataIndex, date: str) -> float:
    return sum(qty * _mark_price(bar_index, code, date) for code, qty in positions.items())


def _mark_price(bar_index: BarDataIndex, code: str, date: str) -> float:
    return bar_index.close_at_or_before(code, date)


def _holding_days(entry_date: object, current_date: object) -> int | None:
    start = pd.to_datetime(entry_date, errors="coerce")
    end = pd.to_datetime(current_date, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return None
    return max(int((end - start).days), 0)


def _execution_priority(order: dict[str, object]) -> int:
    if order.get("status") != "filled":
        return 2
    return 0 if order.get("side") == "sell" else 1


def _cash_limited_buy_qty(qty: float, fill_px: float, cash: float, execution: ExecutionConfig) -> float:
    if fill_px <= 0.0 or cash <= 0.0:
        return 0.0
    affordable_qty = min(qty, cash / fill_px)
    if not execution.allow_fractional_shares:
        affordable_qty = int(affordable_qty // execution.a_share_lot_size) * execution.a_share_lot_size
    return max(affordable_qty, 0.0)
