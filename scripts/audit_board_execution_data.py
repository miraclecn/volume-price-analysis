from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_stock_selector.contracts.board_execution_contract import validate_board_execution_contract


DEFAULT_PREDICTIONS = "outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_predictions.csv"
DEFAULT_BOARD_DB = "outputs/limit_hit_research/board_execution.duckdb"
DEFAULT_OUT_JSON = "outputs/limit_hit_research/reports/board_execution_data_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--board-db", default=DEFAULT_BOARD_DB)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()
    audit = audit_board_execution_data(
        predictions_path=Path(args.predictions),
        board_db=Path(args.board_db),
        top_n=args.top_n,
    )
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


def audit_board_execution_data(*, predictions_path: Path, board_db: Path, top_n: int = 5) -> dict[str, object]:
    if not predictions_path.exists():
        return {"ok": False, "reason": f"missing predictions: {predictions_path}"}
    if not board_db.exists():
        return {"ok": False, "reason": f"missing board execution database: {board_db}"}
    predictions = _top_predictions(pd.read_csv(predictions_path), top_n=top_n)
    con = duckdb.connect(str(board_db), read_only=True)
    try:
        contract = validate_board_execution_contract(con, require_order_book=False, require_fills=False)
        tables = _tables(con)
        summary: dict[str, object] = {
            "ok": bool(contract.ok),
            "reason": "pass" if contract.ok else _contract_reason(contract),
            "contract": {
                "ok": bool(contract.ok),
                "missing_tables": contract.missing_tables,
                "missing_columns": contract.missing_columns,
                "warnings": contract.warnings,
            },
            "prediction_top_rows": int(len(predictions)),
            "prediction_active_days": int(predictions["trade_date"].nunique()) if len(predictions) else 0,
            "tables": sorted(tables),
        }
        if "board_intraday_events" in tables:
            events = con.execute("select cast(trade_date as varchar) as trade_date, cast(code as varchar) as code from board_intraday_events").fetchdf()
            summary["events"] = _coverage_summary(predictions, events)
        if "board_order_book_snapshots" in tables:
            order_book = con.execute("select cast(trade_date as varchar) as trade_date, cast(code as varchar) as code from board_order_book_snapshots").fetchdf()
            summary["order_book"] = _coverage_summary(predictions, order_book)
        if "board_order_fills" in tables:
            fills = con.execute(
                """
                select
                    cast(trade_date as varchar) as trade_date,
                    cast(code as varchar) as code,
                    lower(cast(side as varchar)) as side,
                    filled_qty
                from board_order_fills
                """
            ).fetchdf()
            summary["fills"] = _fill_summary(predictions, fills)
        return summary
    finally:
        con.close()


def _top_predictions(predictions: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    required = {"trade_date", "code", "pred_ret", "pred_win_prob"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {', '.join(missing)}")
    out = predictions.copy()
    out["trade_date"] = out["trade_date"].astype(str)
    out["code"] = out["code"].astype(str)
    return (
        out.sort_values(["trade_date", "pred_ret", "pred_win_prob", "code"], ascending=[True, False, False, True])
        .groupby("trade_date", as_index=False, sort=True)
        .head(top_n)[["trade_date", "code"]]
        .reset_index(drop=True)
    )


def _coverage_summary(top: pd.DataFrame, observed: pd.DataFrame) -> dict[str, object]:
    observed = observed.copy()
    observed["trade_date"] = observed["trade_date"].astype(str)
    observed["code"] = observed["code"].astype(str)
    matched = top.merge(observed.drop_duplicates(), on=["trade_date", "code"], how="inner")
    missing = _missing_top_sample(top, observed)
    active_days = int(top["trade_date"].nunique()) if len(top) else 0
    matched_days = int(matched["trade_date"].nunique()) if len(matched) else 0
    return {
        "rows": int(len(observed)),
        "dates": int(observed["trade_date"].nunique()) if len(observed) else 0,
        "matched_top_rows": int(len(matched)),
        "matched_top_days": matched_days,
        "top_row_coverage": _safe_div(len(matched), len(top)),
        "top_day_coverage": _safe_div(matched_days, active_days),
        "missing_top_sample": missing,
    }


def _fill_summary(top: pd.DataFrame, fills: pd.DataFrame) -> dict[str, object]:
    fills = fills.copy()
    fills["trade_date"] = fills["trade_date"].astype(str)
    fills["code"] = fills["code"].astype(str)
    fills["side"] = fills["side"].astype(str).str.lower()
    fills["filled_qty"] = pd.to_numeric(fills["filled_qty"], errors="coerce").fillna(0.0)
    buy_fills = fills[(fills["side"] == "buy") & (fills["filled_qty"] > 0)].copy()
    base = _coverage_summary(top, buy_fills[["trade_date", "code"]])
    matched = top.merge(buy_fills, on=["trade_date", "code"], how="inner")
    per_day = matched.groupby("trade_date")["code"].nunique() if len(matched) else pd.Series(dtype=float)
    base.update(
        {
            "buy_fill_rows": int(len(buy_fills)),
            "avg_matched_fills_per_top_day": float(per_day.mean()) if len(per_day) else 0.0,
            "days_with_at_least_2_matched_fills": int((per_day >= 2).sum()) if len(per_day) else 0,
            "days_with_at_least_2_matched_fills_rate": _safe_div(int((per_day >= 2).sum()) if len(per_day) else 0, int(top["trade_date"].nunique()) if len(top) else 0),
        }
    )
    return base


def _tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    return {
        row[0]
        for row in con.execute(
            "select table_name from information_schema.tables where table_schema = 'main'"
        ).fetchall()
    }


def _missing_top_sample(top: pd.DataFrame, observed: pd.DataFrame, *, limit: int = 20) -> list[dict[str, str]]:
    observed_keys = observed[["trade_date", "code"]].drop_duplicates().copy()
    missing = top.merge(observed_keys, on=["trade_date", "code"], how="left", indicator=True)
    missing = missing[missing["_merge"].eq("left_only")][["trade_date", "code"]].head(limit)
    return [{"trade_date": str(row.trade_date), "code": str(row.code)} for row in missing.itertuples(index=False)]


def _contract_reason(contract) -> str:
    messages = []
    if contract.missing_tables:
        messages.append(f"missing tables: {', '.join(contract.missing_tables)}")
    for table, cols in contract.missing_columns.items():
        messages.append(f"missing columns in {table}: {', '.join(cols)}")
    return "; ".join(messages) if messages else "contract warnings only"


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


if __name__ == "__main__":
    main()
