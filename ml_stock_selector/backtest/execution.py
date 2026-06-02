from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ORDER_COLUMNS = [
    "run_id",
    "sim_date",
    "decision_date",
    "code",
    "side",
    "qty",
    "target_weight",
    "order_px_ref",
    "fill_px",
    "status",
    "reason",
    "entry_date",
    "exit_date",
    "holding_days",
    "entry_trade_score",
    "exit_trade_score",
    "entry_reason",
    "exit_reason",
    "sell_blocked_reason",
]


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
    decision_date: str | None = None,
) -> pd.DataFrame:
    orders = []
    current_qty = _current_qty_by_code(current_positions)
    if targets.empty and decision_date is None:
        return pd.DataFrame(columns=ORDER_COLUMNS)
    if decision_date is None:
        decision_date = str(targets["trade_date"].iloc[0])
    bars_sorted = bars if bars.attrs.get("_vpa_sorted_by_code_date") else bars.sort_values(["code", "trade_date"])
    target_weights = {str(row.code): float(row.target_weight) for row in targets.itertuples(index=False)} if not targets.empty else {}
    target_meta = _rows_by_code(targets)
    position_meta = _rows_by_code(current_positions)
    codes = sorted(set(target_weights) | {code for code, qty in current_qty.items() if abs(qty) > 1e-12})
    for code in codes:
        target_weight = target_weights.get(code, 0.0)
        future = bars_sorted[(bars_sorted["code"] == code) & (bars_sorted["trade_date"] > decision_date)]
        if future.empty:
            side = "sell" if target_weight == 0.0 and current_qty.get(code, 0.0) > 0.0 else "buy"
            orders.append(_order_row(decision_date, code, target_weight, None, side, 0.0, None, "rejected", "no_next_bar", config, target_meta.get(code), position_meta.get(code)))
            continue
        bar = future.iloc[0]
        reference_px = float(bar["open"])
        desired_qty = (nav or 0.0) * target_weight / reference_px if reference_px and target_weight > 0.0 else 0.0
        delta_qty = desired_qty - current_qty.get(code, 0.0)
        if abs(delta_qty) <= 1e-9:
            continue
        side = "buy" if delta_qty > 0.0 else "sell"
        reason = _reason(bar, side)
        status = "filled" if reason is None else "rejected"
        qty = abs(delta_qty) if status == "filled" else 0.0
        fill_px = _fill_price(bar, side, config) if status == "filled" else None
        orders.append(_order_row(decision_date, code, target_weight, bar, side, qty, fill_px, status, reason, config, target_meta.get(code), position_meta.get(code)))
    out = pd.DataFrame(orders, columns=ORDER_COLUMNS)
    if not out.empty:
        assert_no_t0_fills(out)
    return out


def assert_no_t0_fills(orders: pd.DataFrame) -> None:
    bad = orders[(orders["status"] == "filled") & (orders["sim_date"] <= orders["decision_date"])]
    if not bad.empty:
        raise ValueError("T+1 execution violation: filled order has sim_date <= decision_date")


def _current_qty_by_code(current_positions: pd.DataFrame) -> dict[str, float]:
    if current_positions.empty or "code" not in current_positions:
        return {}
    qty_col = "position_qty" if "position_qty" in current_positions else "qty"
    if qty_col not in current_positions:
        return {}
    return {str(row.code): float(getattr(row, qty_col)) for row in current_positions.itertuples(index=False)}


def _rows_by_code(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty or "code" not in frame:
        return {}
    return {
        str(row["code"]): row
        for _, row in frame.drop_duplicates("code", keep="last").iterrows()
    }


def _fill_price(bar: pd.Series, side: str, config: ExecutionConfig) -> float:
    sign = 1 if side == "buy" else -1
    return float(bar["open"]) * (1 + sign * config.slippage_bps / 10000.0)


def _order_row(
    decision_date: str,
    code: str,
    target_weight: float,
    bar,
    side: str,
    qty: float,
    fill_px: float | None,
    status: str,
    reason: str | None,
    config: ExecutionConfig,
    target_row: pd.Series | None = None,
    position_row: pd.Series | None = None,
) -> dict[str, object]:
    sim_date = bar["trade_date"] if bar is not None else decision_date
    if status == "filled" and not config.allow_fractional_shares:
        qty = int(qty // config.a_share_lot_size) * config.a_share_lot_size
    entry_date = _first_present(position_row, "entry_date") if side == "sell" else None
    entry_price = _first_present(position_row, "entry_price") if side == "sell" else None
    entry_trade_score = _first_present(position_row, "entry_trade_score") if side == "sell" else _first_present(target_row, "trade_score", "trade_score_v2")
    exit_trade_score = _first_present(target_row, "trade_score", "trade_score_v2") if side == "sell" else None
    exit_reason = _first_present(target_row, "exit_reason") if side == "sell" else None
    holding_days = _holding_days(entry_date, sim_date) if side == "sell" and entry_date is not None else None
    sell_blocked_reason = reason if side == "sell" and status != "filled" else _first_present(target_row, "sell_blocked_reason")
    return {
        "run_id": None,
        "sim_date": sim_date,
        "decision_date": decision_date,
        "code": code,
        "side": side,
        "qty": qty,
        "target_weight": target_weight,
        "order_px_ref": config.execution_price,
        "fill_px": fill_px,
        "status": status,
        "reason": reason,
        "entry_date": entry_date,
        "exit_date": sim_date if side == "sell" and status == "filled" else None,
        "holding_days": holding_days,
        "entry_trade_score": entry_trade_score,
        "exit_trade_score": exit_trade_score,
        "entry_reason": _first_present(position_row, "entry_reason") if side == "sell" else _first_present(target_row, "entry_reason"),
        "exit_reason": exit_reason,
        "sell_blocked_reason": sell_blocked_reason,
    }


def _reason(bar: pd.Series, side: str) -> str | None:
    if bool(bar.get("is_paused", False)):
        return "paused"
    if side == "buy" and float(bar["open"]) >= float(bar["limit_up"]):
        return "limit_up"
    if side == "sell" and float(bar["open"]) <= float(bar["limit_down"]):
        return "limit_down"
    return None


def _first_present(row: pd.Series | None, *columns: str) -> object | None:
    if row is None:
        return None
    for column in columns:
        if column in row and not pd.isna(row[column]):
            return row[column]
    return None


def _holding_days(entry_date: object, exit_date: object) -> int | None:
    start = pd.to_datetime(entry_date, errors="coerce")
    end = pd.to_datetime(exit_date, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return None
    return max(int((end - start).days), 0)
