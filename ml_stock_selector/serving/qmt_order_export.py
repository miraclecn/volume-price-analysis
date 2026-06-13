from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

SCHEMA_VERSION = "qmt_ml_order.v1"
DEFAULT_QMT_ORDER_JSON = Path("outputs/ml/live_sim/qmt_orders/orders.json")


def export_qmt_orders(
    con: duckdb.DuckDBPyConnection,
    account_id: str,
    decision_date: str,
    execution_date: str | None = None,
    output_json: Path | str = DEFAULT_QMT_ORDER_JSON,
    copy_dir: Path | str | None = None,
    generated_at: str | None = None,
) -> Path:
    output_path = Path(output_json)
    orders = _load_planned_orders(con, account_id, decision_date)
    payload = _payload_from_orders(account_id, decision_date, orders, generated_at or _now(), execution_date=execution_date)
    _write_json_atomic(output_path, payload)
    if copy_dir is not None:
        _validate_copy_dir(copy_dir)
        copy_path = Path(copy_dir) / output_path.name
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, copy_path)
    return output_path


def _load_planned_orders(con: duckdb.DuckDBPyConnection, account_id: str, decision_date: str) -> pd.DataFrame:
    return con.execute(
        """
        select *
        from live_sim_planned_orders
        where account_id = ?
          and decision_date = ?
          and status = 'planned'
          and lower(side) in ('buy', 'sell')
        order by case lower(side) when 'buy' then 0 when 'sell' then 1 else 2 end, code
        """,
        [account_id, decision_date],
    ).fetchdf()


def _payload_from_orders(
    account_id: str,
    decision_date: str,
    orders: pd.DataFrame,
    generated_at: str,
    execution_date: str | None = None,
) -> dict[str, Any]:
    execution_date = execution_date or _execution_date(orders)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "live_sim_daily",
        "account_id": account_id,
        "decision_date": decision_date,
        "execution_date": execution_date,
        "generated_at": generated_at,
        "orders": [_order_record(account_id, decision_date, row) for row in orders.to_dict("records")],
    }


def _order_record(account_id: str, decision_date: str, row: dict[str, Any]) -> dict[str, Any]:
    side = str(row.get("side", "")).lower()
    execution_date = _clean(row.get("execution_date"))
    code = str(row.get("code"))
    return {
        "client_order_id": f"{account_id}_{decision_date.replace('-', '')}_{code.replace('.', '_')}_{side}",
        "side": side,
        "code": code,
        "qty": _qty(row.get("estimated_qty")),
        "reference_price": _number(row.get("estimated_price")),
        "reference_price_type": "decision_close",
        "buy_date": execution_date if side == "buy" else None,
        "sell_date": execution_date if side == "sell" else None,
        "target_weight": _number(row.get("target_weight")),
        "target_value": _number(row.get("target_value")),
        "status": _clean(row.get("status")) or "planned",
    }


def _execution_date(orders: pd.DataFrame) -> str | None:
    if orders.empty or "execution_date" not in orders:
        return None
    values = [str(value)[:10] for value in orders["execution_date"].dropna().tolist()]
    return values[0] if values else None


def _qty(value: Any) -> int | float | None:
    number = _number(value)
    if number is None:
        return None
    if float(number).is_integer():
        return int(number)
    return number


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return float(value)


def _clean(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)[:10] if "date" in str(type(value)).lower() else str(value)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _validate_copy_dir(copy_dir: Path | str) -> None:
    text = str(copy_dir)
    if os.name != "nt" and text.startswith("\\\\"):
        raise ValueError(
            "UNC copy paths are only directly usable from Windows. "
            "On Linux/WSL, mount the SMB share first and pass the mounted path."
        )


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
