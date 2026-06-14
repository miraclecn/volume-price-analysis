from __future__ import annotations

import math
from typing import Mapping

import pandas as pd


def build_live_target_positions(
    targets: pd.DataFrame,
    allocation: pd.DataFrame,
    *,
    trade_date: str,
    account_id: str,
    account_nav: float,
    generated_at: str,
) -> pd.DataFrame:
    columns = [
        "trade_date",
        "account_id",
        "strategy_id",
        "code",
        "target_weight",
        "target_value",
        "source_bundle_id",
        "source_sleeve",
        "score_version",
        "reason",
        "generated_at",
    ]
    if targets.empty:
        return pd.DataFrame(columns=columns)
    prepared_targets = targets.copy()
    prepared_targets["trade_date"] = prepared_targets.get("trade_date", trade_date)
    prepared_targets = prepared_targets[prepared_targets["trade_date"].astype(str) == str(trade_date)]
    if prepared_targets.empty:
        return pd.DataFrame(columns=columns)
    prepared_targets["strategy_id"] = prepared_targets.get("portfolio_id", prepared_targets.get("strategy_id", "unknown_strategy")).astype(str)
    prepared_targets["score_version"] = prepared_targets.get("score_version", "").astype(str)

    alloc = allocation.copy()
    if alloc.empty:
        alloc = pd.DataFrame(columns=["trade_date", "strategy_id", "score_version", "sleeve", "bundle_id", "final_weight"])
    alloc["trade_date"] = alloc.get("trade_date", trade_date)
    alloc = alloc[alloc["trade_date"].astype(str) == str(trade_date)]
    merged = prepared_targets.merge(
        alloc[["strategy_id", "score_version", "sleeve", "bundle_id", "final_weight"]],
        on=["strategy_id", "score_version"],
        how="left",
    )
    merged["final_weight"] = pd.to_numeric(merged["final_weight"], errors="coerce").fillna(1.0)
    merged["target_weight"] = (
        pd.to_numeric(merged.get("target_weight", pd.Series(0.0, index=merged.index)), errors="coerce").fillna(0.0)
        * merged["final_weight"]
    ).round(12)
    merged["target_value"] = (merged["target_weight"] * float(account_nav)).round(12)
    merged["reason"] = _first_available(merged, ["signal_action", "entry_reason", "reason"], default="selected")
    out = pd.DataFrame(
        {
            "trade_date": trade_date,
            "account_id": account_id,
            "strategy_id": merged["strategy_id"],
            "code": merged["code"].astype(str),
            "target_weight": merged["target_weight"],
            "target_value": merged["target_value"],
            "source_bundle_id": merged.get("bundle_id", pd.Series("", index=merged.index)).fillna("").astype(str),
            "source_sleeve": merged.get("sleeve", pd.Series("", index=merged.index)).fillna("").astype(str),
            "score_version": merged["score_version"],
            "reason": merged["reason"].fillna("selected").astype(str),
            "generated_at": generated_at,
        }
    )
    return out[columns]


def build_live_orders(
    target_positions: pd.DataFrame,
    *,
    execution_date: str,
    prices: Mapping[str, float],
    generated_at: str,
) -> pd.DataFrame:
    columns = [
        "order_id",
        "trade_date",
        "account_id",
        "strategy_id",
        "code",
        "side",
        "order_qty",
        "order_price",
        "status",
        "block_reason",
        "created_at",
        "submitted_at",
        "updated_at",
    ]
    if target_positions.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for row in target_positions.to_dict("records"):
        target_value = float(row.get("target_value") or 0.0)
        code = str(row["code"])
        price = float(prices.get(code, 0.0) or 0.0)
        side = "buy" if target_value > 0.0 else "sell"
        blocked = price <= 0.0
        qty = 0.0 if blocked else math.floor(abs(target_value) / price / 100.0) * 100.0
        order_id = _order_id(str(row["account_id"]), execution_date, str(row["strategy_id"]), code, side)
        rows.append(
            {
                "order_id": order_id,
                "trade_date": execution_date,
                "account_id": row["account_id"],
                "strategy_id": row["strategy_id"],
                "code": code,
                "side": side,
                "order_qty": qty,
                "order_price": price if price > 0.0 else None,
                "status": "blocked" if blocked or qty <= 0.0 else "created",
                "block_reason": "missing_price" if blocked else ("zero_qty" if qty <= 0.0 else None),
                "created_at": generated_at,
                "submitted_at": None,
                "updated_at": generated_at,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def record_live_fills(
    orders: pd.DataFrame,
    *,
    fill_prices: Mapping[str, float],
    fill_time: str,
    commission_rate: float = 0.0003,
    tax_rate: float = 0.001,
) -> pd.DataFrame:
    columns = [
        "fill_id",
        "order_id",
        "trade_date",
        "code",
        "side",
        "fill_qty",
        "fill_price",
        "fill_time",
        "commission",
        "tax",
        "slippage_bps",
    ]
    if orders.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for row in orders[orders["status"].astype(str).isin(["created", "submitted"])].to_dict("records"):
        code = str(row["code"])
        fill_price = float(fill_prices.get(code, row.get("order_price") or 0.0) or 0.0)
        order_price = float(row.get("order_price") or 0.0)
        qty = float(row.get("order_qty") or 0.0)
        traded_value = qty * fill_price
        side = str(row["side"])
        slippage_bps = 0.0 if order_price <= 0.0 else round((fill_price - order_price) / order_price * 10000.0, 6)
        rows.append(
            {
                "fill_id": f"{row['order_id']}_fill",
                "order_id": row["order_id"],
                "trade_date": row["trade_date"],
                "code": code,
                "side": side,
                "fill_qty": qty,
                "fill_price": fill_price,
                "fill_time": fill_time,
                "commission": round(traded_value * commission_rate, 12),
                "tax": round(traded_value * tax_rate, 12) if side == "sell" else 0.0,
                "slippage_bps": slippage_bps,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_risk_logs(
    target_positions: pd.DataFrame,
    *,
    trade_date: str,
    account_id: str,
    strategy_id: str,
    generated_at: str,
) -> pd.DataFrame:
    has_targets = not target_positions.empty and pd.to_numeric(
        target_positions.get("target_weight", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0).abs().sum() > 0.0
    return pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "account_id": account_id,
                "strategy_id": strategy_id,
                "check_name": "target_positions_present",
                "severity": "info" if has_targets else "warning",
                "passed": bool(has_targets),
                "action": "allow" if has_targets else "no_order",
                "reason": "target positions generated" if has_targets else "no target positions generated",
                "created_at": generated_at,
            }
        ]
    ).astype({"passed": object})


def _first_available(frame: pd.DataFrame, columns: list[str], *, default: str) -> pd.Series:
    output = pd.Series(default, index=frame.index, dtype=object)
    for column in columns:
        if column in frame:
            values = frame[column].fillna("").astype(str)
            output = output.where(output.astype(str) != default, values.where(values != "", default))
    return output


def _order_id(account_id: str, trade_date: str, strategy_id: str, code: str, side: str) -> str:
    return f"{account_id}_{trade_date.replace('-', '')}_{strategy_id}_{code.replace('.', '_')}_{side}"
