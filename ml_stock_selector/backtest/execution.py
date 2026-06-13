from __future__ import annotations

from bisect import bisect_right
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
    "entry_price",
    "exit_date",
    "holding_days",
    "entry_trade_score",
    "exit_trade_score",
    "entry_abs_rank_pct",
    "entry_risk_rank_pct",
    "entry_reason",
    "exit_reason",
    "sell_blocked_reason",
    "strategy_id",
    "realized_pnl",
]


@dataclass(frozen=True)
class ExecutionConfig:
    execution_price: str = "next_open"
    slippage_bps: float = 5.0
    commission_bps: float = 3.0
    stamp_duty_bps: float = 5.0
    a_share_lot_size: int = 100
    allow_fractional_shares: bool = True


@dataclass(frozen=True)
class _CodeBars:
    date_keys: list[str]
    trade_dates: list[object]
    opens: list[float]
    closes: list[float]
    limit_ups: list[float]
    limit_downs: list[float]
    paused: list[bool]


@dataclass(frozen=True)
class BarDataIndex:
    bars_by_code: dict[str, _CodeBars]

    @classmethod
    def from_bars(cls, bars: pd.DataFrame) -> "BarDataIndex":
        if bars.empty:
            return cls({})
        bars_sorted = bars if bars.attrs.get("_vpa_sorted_by_code_date") else bars.sort_values(["code", "trade_date"])
        bars_by_code: dict[str, _CodeBars] = {}
        for code, frame in bars_sorted.groupby("code", sort=False):
            code_key = str(code)
            trade_dates = frame["trade_date"].tolist()
            bars_by_code[code_key] = _CodeBars(
                date_keys=[_date_key(value) for value in trade_dates],
                trade_dates=trade_dates,
                opens=_float_values(frame, "open"),
                closes=_float_values(frame, "close"),
                limit_ups=_float_values(frame, "limit_up"),
                limit_downs=_float_values(frame, "limit_down"),
                paused=_bool_values(frame, "is_paused"),
            )
        return cls(bars_by_code)

    def next_bar(self, code: str, decision_date: object) -> dict[str, object] | None:
        code_key = str(code)
        code_bars = self.bars_by_code.get(code_key)
        if code_bars is None:
            return None
        pos = bisect_right(code_bars.date_keys, _date_key(decision_date))
        if pos >= len(code_bars.date_keys):
            return None
        return {
            "trade_date": code_bars.trade_dates[pos],
            "open": code_bars.opens[pos],
            "close": code_bars.closes[pos],
            "limit_up": code_bars.limit_ups[pos],
            "limit_down": code_bars.limit_downs[pos],
            "is_paused": code_bars.paused[pos],
        }

    def close_at_or_before(self, code: str, date: object) -> float:
        code_key = str(code)
        code_bars = self.bars_by_code.get(code_key)
        if code_bars is None:
            return 0.0
        pos = bisect_right(code_bars.date_keys, _date_key(date)) - 1
        if pos < 0:
            return 0.0
        return code_bars.closes[pos]


def simulate_rebalance_orders(
    targets: pd.DataFrame,
    bars: pd.DataFrame,
    current_positions: pd.DataFrame,
    nav: float,
    config: ExecutionConfig,
    decision_date: str | None = None,
    bar_index: BarDataIndex | None = None,
) -> pd.DataFrame:
    orders = []
    current_qty = _current_qty_by_code(current_positions)
    if targets.empty and decision_date is None:
        return pd.DataFrame(columns=ORDER_COLUMNS)
    if decision_date is None:
        decision_date = str(targets["trade_date"].iloc[0])
    bars_sorted = None if bar_index is not None else (bars if bars.attrs.get("_vpa_sorted_by_code_date") else bars.sort_values(["code", "trade_date"]))
    bar_index = bar_index or BarDataIndex.from_bars(bars_sorted)
    target_weights = {str(row.code): float(row.target_weight) for row in targets.itertuples(index=False)} if not targets.empty else {}
    target_meta = _rows_by_code(targets)
    position_meta = _rows_by_code(current_positions)
    codes = sorted(set(target_weights) | {code for code, qty in current_qty.items() if abs(qty) > 1e-12})
    for code in codes:
        target_weight = target_weights.get(code, 0.0)
        target_row = target_meta.get(code)
        current_code_qty = current_qty.get(code, 0.0)
        if _keeps_existing_quantity(target_row, current_code_qty):
            continue
        bar = bar_index.next_bar(code, decision_date)
        if bar is None:
            side = "sell" if target_weight == 0.0 and current_code_qty > 0.0 else "buy"
            orders.append(_order_row(decision_date, code, target_weight, None, side, 0.0, None, "rejected", "no_next_bar", config, target_row, position_meta.get(code)))
            continue
        reference_px = float(bar["open"])
        desired_qty = (nav or 0.0) * target_weight / reference_px if reference_px and target_weight > 0.0 else 0.0
        delta_qty = desired_qty - current_code_qty
        if abs(delta_qty) <= 1e-9:
            continue
        side = "buy" if delta_qty > 0.0 else "sell"
        reason = _reason(bar, side)
        status = "filled" if reason is None else "rejected"
        qty = abs(delta_qty) if status == "filled" else 0.0
        fill_px = _fill_price(bar, side, config) if status == "filled" else None
        orders.append(_order_row(decision_date, code, target_weight, bar, side, qty, fill_px, status, reason, config, target_row, position_meta.get(code)))
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


def _date_key(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def _float_values(frame: pd.DataFrame, column: str) -> list[float]:
    if column not in frame:
        return [0.0] * len(frame)
    values = []
    for value in frame[column].tolist():
        values.append(0.0 if pd.isna(value) else float(value))
    return values


def _bool_values(frame: pd.DataFrame, column: str) -> list[bool]:
    if column not in frame:
        return [False] * len(frame)
    values = []
    for value in frame[column].tolist():
        values.append(False if pd.isna(value) else bool(value))
    return values


def _rows_by_code(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty or "code" not in frame:
        return {}
    return {
        str(row["code"]): row
        for _, row in frame.drop_duplicates("code", keep="last").iterrows()
    }


def _keeps_existing_quantity(target_row: pd.Series | None, current_qty: float) -> bool:
    if target_row is None or abs(current_qty) <= 1e-12:
        return False
    action = _first_present(target_row, "signal_action")
    return str(action).lower() in {"hold", "sell_blocked"}


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
    entry_abs_rank_pct = _first_present(position_row, "entry_abs_rank_pct") if side == "sell" else _first_present(target_row, "entry_abs_rank_pct", "absolute_rank_pct")
    entry_risk_rank_pct = _first_present(position_row, "entry_risk_rank_pct") if side == "sell" else _first_present(target_row, "entry_risk_rank_pct", "risk_rank_pct")
    exit_reason = _first_present(target_row, "exit_reason") if side == "sell" else None
    holding_days = _first_present(position_row, "holding_days") if side == "sell" else None
    if holding_days is None and side == "sell" and entry_date is not None:
        holding_days = _holding_days(entry_date, sim_date)
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
        "entry_price": entry_price,
        "exit_date": sim_date if side == "sell" and status == "filled" else None,
        "holding_days": holding_days,
        "entry_trade_score": entry_trade_score,
        "exit_trade_score": exit_trade_score,
        "entry_abs_rank_pct": entry_abs_rank_pct,
        "entry_risk_rank_pct": entry_risk_rank_pct,
        "entry_reason": _first_present(position_row, "entry_reason") if side == "sell" else _first_present(target_row, "entry_reason"),
        "exit_reason": exit_reason,
        "sell_blocked_reason": sell_blocked_reason,
        "strategy_id": _first_present(target_row, "portfolio_id", "strategy_id"),
        "realized_pnl": None,
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
