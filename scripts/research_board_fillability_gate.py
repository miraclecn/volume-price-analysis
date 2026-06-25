from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.research_board_overnight_model import _portfolio_metrics


DEFAULT_PREDICTIONS = "outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_predictions.csv"
DEFAULT_BOARD_DB = "outputs/limit_hit_research/board_execution.duckdb"
DEFAULT_OUT_DIR = "outputs/limit_hit_research/reports/board_fillability_gate_2020_2025"


@dataclass(frozen=True)
class FillabilityGateConfig:
    max_candidates: int = 5
    benchmark_fills: int = 2
    name_weight: float = 0.01
    total_exposure_cap: float = 0.05
    benchmark_extra_slippage_bps: float = 50.0
    min_avg_fills_per_active_day: float = 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--board-db", default=DEFAULT_BOARD_DB)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--benchmark-fills", type=int, default=2)
    parser.add_argument("--name-weight", type=float, default=0.01)
    parser.add_argument("--total-exposure-cap", type=float, default=0.05)
    parser.add_argument("--benchmark-extra-slippage-bps", type=float, default=50.0)
    parser.add_argument("--min-avg-fills-per-active-day", type=float, default=2.0)
    args = parser.parse_args()
    config = FillabilityGateConfig(
        max_candidates=args.max_candidates,
        benchmark_fills=args.benchmark_fills,
        name_weight=args.name_weight,
        total_exposure_cap=args.total_exposure_cap,
        benchmark_extra_slippage_bps=args.benchmark_extra_slippage_bps,
        min_avg_fills_per_active_day=args.min_avg_fills_per_active_day,
    )
    result = run_fillability_gate(
        predictions_path=Path(args.predictions),
        board_db=Path(args.board_db),
        out_dir=Path(args.out_dir),
        config=config,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


def run_fillability_gate(
    *,
    predictions_path: Path,
    board_db: Path,
    out_dir: Path,
    config: FillabilityGateConfig | None = None,
) -> dict[str, object]:
    config = config or FillabilityGateConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(predictions_path)
    fills_result = load_board_fills(board_db)
    if not fills_result["ok"]:
        summary = {
            "ok": False,
            "reason": fills_result["reason"],
            "predictions": str(predictions_path),
            "board_db": str(board_db),
            "config": asdict(config),
        }
        _write_outputs(out_dir, summary, pd.DataFrame(), pd.DataFrame())
        return {"summary": summary, "daily": pd.DataFrame(), "fills": pd.DataFrame()}

    result = evaluate_fillability(predictions, fills_result["fills"], config)
    summary = {
        **result["summary"],
        "predictions": str(predictions_path),
        "board_db": str(board_db),
        "config": asdict(config),
    }
    _write_outputs(out_dir, summary, result["daily"], result["filled_orders"])
    return {"summary": summary, "daily": result["daily"], "fills": result["filled_orders"]}


def load_board_fills(board_db: Path) -> dict[str, object]:
    if not board_db.exists():
        return {"ok": False, "reason": f"missing board execution database: {board_db}", "fills": pd.DataFrame()}
    con = duckdb.connect(str(board_db), read_only=True)
    try:
        tables = {
            row[0]
            for row in con.execute(
                "select table_name from information_schema.tables where table_schema = 'main'"
            ).fetchall()
        }
        if "board_order_fills" not in tables:
            return {"ok": False, "reason": "missing board_order_fills", "fills": pd.DataFrame()}
        fills = con.execute(
            """
            select
                cast(trade_date as varchar) as trade_date,
                cast(code as varchar) as code,
                cast(signal_time as varchar) as signal_time,
                cast(order_time as varchar) as order_time,
                lower(cast(side as varchar)) as side,
                order_price,
                order_qty,
                filled_qty,
                avg_fill_price,
                status
            from board_order_fills
            """
        ).fetchdf()
    finally:
        con.close()
    return {"ok": True, "reason": "", "fills": fills}


def evaluate_fillability(
    predictions: pd.DataFrame,
    fills: pd.DataFrame,
    config: FillabilityGateConfig,
) -> dict[str, object]:
    top = _top_candidates(predictions, config.max_candidates)
    buy_fills = _normalize_buy_fills(fills)
    daily_rows = []
    filled_order_rows = []
    actual_day_returns = []
    benchmark_day_returns = []
    ideal_day_returns = []
    for trade_date, day in top.groupby("trade_date", sort=True):
        candidates = day.sort_values(["pred_ret", "pred_win_prob", "code"], ascending=[False, False, True]).copy()
        day_fills = buy_fills[buy_fills["trade_date"].astype(str).eq(str(trade_date))]
        filled = candidates.merge(day_fills, on=["trade_date", "code"], how="inner")
        if not filled.empty:
            filled["filled_return_net"] = _entry_adjusted_returns(filled)
        actual_day_return = _weighted_day_return(filled, "filled_return_net", config)
        benchmark = _adverse_benchmark(candidates, config)
        ideal = candidates.head(min(config.benchmark_fills, len(candidates))).copy()
        ideal["ideal_return_net"] = pd.to_numeric(ideal["target_ret_net"], errors="coerce").fillna(0.0)
        ideal_day_return = _weighted_day_return(ideal, "ideal_return_net", config)

        actual_day_returns.append(actual_day_return)
        benchmark_day_returns.append(_weighted_day_return(benchmark, "benchmark_return_net", config))
        ideal_day_returns.append(ideal_day_return)
        daily_rows.append(
            {
                "trade_date": trade_date,
                "candidate_count": int(len(candidates)),
                "filled_count": int(len(filled)),
                "candidate_codes": ",".join(candidates["code"].astype(str).tolist()),
                "filled_codes": ",".join(filled["code"].astype(str).tolist()) if not filled.empty else "",
                "filled_return_mean": _safe_mean(filled.get("filled_return_net", pd.Series(dtype=float))),
                "ideal_fill_return_mean": _safe_mean(ideal["ideal_return_net"]),
                "adverse_benchmark_return_mean": _safe_mean(benchmark["benchmark_return_net"]),
                "filled_day_return": actual_day_return,
                "ideal_day_return": ideal_day_return,
                "adverse_benchmark_day_return": benchmark_day_returns[-1],
            }
        )
        if not filled.empty:
            for row in filled.itertuples(index=False):
                filled_order_rows.append(
                    {
                        "trade_date": row.trade_date,
                        "code": row.code,
                        "filled_qty": float(row.filled_qty),
                        "order_price": _nullable_float(row.order_price),
                        "avg_fill_price": _nullable_float(row.avg_fill_price),
                        "target_ret_net": float(row.target_ret_net),
                        "filled_return_net": float(row.filled_return_net),
                        "pred_ret": float(row.pred_ret),
                        "pred_win_prob": float(row.pred_win_prob),
                    }
                )

    daily = pd.DataFrame(daily_rows)
    filled_orders = pd.DataFrame(filled_order_rows)
    active_days = int(len(daily))
    actual_nav = _nav_from_day_returns(daily["trade_date"].tolist(), actual_day_returns)
    benchmark_nav = _nav_from_day_returns(daily["trade_date"].tolist(), benchmark_day_returns)
    ideal_nav = _nav_from_day_returns(daily["trade_date"].tolist(), ideal_day_returns)
    avg_fills = float(daily["filled_count"].mean()) if active_days else 0.0
    avg_filled_return = _safe_mean(filled_orders.get("filled_return_net", pd.Series(dtype=float)))
    avg_benchmark_return = _safe_mean(daily["adverse_benchmark_return_mean"]) if active_days else 0.0
    fill_count_ok = avg_fills >= config.min_avg_fills_per_active_day
    quality_ok = avg_filled_return >= avg_benchmark_return if len(filled_orders) else False
    summary = {
        "ok": bool(fill_count_ok and quality_ok),
        "reason": "pass" if fill_count_ok and quality_ok else _failure_reason(fill_count_ok, quality_ok, len(filled_orders)),
        "active_days": active_days,
        "candidate_count": int(daily["candidate_count"].sum()) if active_days else 0,
        "filled_count": int(daily["filled_count"].sum()) if active_days else 0,
        "avg_fills_per_active_day": avg_fills,
        "days_with_min_fills_rate": float((daily["filled_count"] >= config.benchmark_fills).mean()) if active_days else 0.0,
        "avg_filled_return": avg_filled_return,
        "avg_adverse_benchmark_return": avg_benchmark_return,
        "fill_count_ok": bool(fill_count_ok),
        "quality_ok": bool(quality_ok),
        "actual_metrics": _portfolio_metrics(actual_nav),
        "adverse_benchmark_metrics": _portfolio_metrics(benchmark_nav),
        "ideal_benchmark_metrics": _portfolio_metrics(ideal_nav),
    }
    return {"summary": summary, "daily": daily, "filled_orders": filled_orders}


def _top_candidates(predictions: pd.DataFrame, max_candidates: int) -> pd.DataFrame:
    required = {"trade_date", "code", "target_ret_net", "pred_ret", "pred_win_prob"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {', '.join(missing)}")
    out = predictions.copy()
    out["trade_date"] = out["trade_date"].astype(str)
    out["code"] = out["code"].astype(str)
    return (
        out.sort_values(["trade_date", "pred_ret", "pred_win_prob", "code"], ascending=[True, False, False, True])
        .groupby("trade_date", as_index=False, sort=True)
        .head(max_candidates)
        .reset_index(drop=True)
    )


def _normalize_buy_fills(fills: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return fills.copy()
    out = fills.copy()
    out["trade_date"] = out["trade_date"].astype(str)
    out["code"] = out["code"].astype(str)
    out["side"] = out["side"].astype(str).str.lower()
    out["filled_qty"] = pd.to_numeric(out["filled_qty"], errors="coerce").fillna(0.0)
    out["order_price"] = pd.to_numeric(out.get("order_price"), errors="coerce")
    out["avg_fill_price"] = pd.to_numeric(out.get("avg_fill_price"), errors="coerce")
    out = out[(out["side"] == "buy") & (out["filled_qty"] > 0)].copy()
    if out.empty:
        return out
    grouped = []
    for (trade_date, code), group in out.groupby(["trade_date", "code"], sort=True):
        qty = float(group["filled_qty"].sum())
        avg_fill = _weighted_price(group, "avg_fill_price", qty)
        order_price = _weighted_price(group, "order_price", qty)
        grouped.append(
            {
                "trade_date": trade_date,
                "code": code,
                "filled_qty": qty,
                "order_price": order_price,
                "avg_fill_price": avg_fill,
            }
        )
    return pd.DataFrame(grouped)


def _weighted_price(group: pd.DataFrame, column: str, qty: float) -> float:
    values = pd.to_numeric(group[column], errors="coerce")
    qtys = pd.to_numeric(group["filled_qty"], errors="coerce").fillna(0.0)
    usable = values.notna() & (values > 0) & (qtys > 0)
    if not usable.any() or qty <= 0:
        return np.nan
    return float((values[usable] * qtys[usable]).sum() / qtys[usable].sum())


def _entry_adjusted_returns(filled: pd.DataFrame) -> pd.Series:
    base = pd.to_numeric(filled["target_ret_net"], errors="coerce").fillna(0.0)
    order_price = pd.to_numeric(filled.get("order_price"), errors="coerce")
    avg_fill_price = pd.to_numeric(filled.get("avg_fill_price"), errors="coerce")
    usable = order_price.notna() & avg_fill_price.notna() & (order_price > 0) & (avg_fill_price > 0)
    adjusted = base.copy()
    adjusted.loc[usable] = (1.0 + base.loc[usable]) * (order_price.loc[usable] / avg_fill_price.loc[usable]) - 1.0
    return adjusted


def _adverse_benchmark(candidates: pd.DataFrame, config: FillabilityGateConfig) -> pd.DataFrame:
    n = min(config.benchmark_fills, len(candidates))
    out = candidates.sort_values(["target_ret_net", "pred_ret", "code"], ascending=[True, False, True]).head(n).copy()
    out["benchmark_return_net"] = pd.to_numeric(out["target_ret_net"], errors="coerce").fillna(0.0) - config.benchmark_extra_slippage_bps / 10000.0
    return out


def _weighted_day_return(frame: pd.DataFrame, return_col: str, config: FillabilityGateConfig) -> float:
    if frame.empty:
        return 0.0
    total_weight = min(float(config.total_exposure_cap), float(config.name_weight) * len(frame))
    weight = total_weight / len(frame)
    return float((pd.to_numeric(frame[return_col], errors="coerce").fillna(0.0) * weight).sum())


def _nav_from_day_returns(dates: list[object], returns: list[float], *, initial_nav: float = 1_000_000.0) -> pd.DataFrame:
    nav = float(initial_nav)
    peak = nav
    rows = []
    for date, ret in zip(dates, returns):
        nav *= 1.0 + float(ret)
        peak = max(peak, nav)
        rows.append({"trade_date": date, "nav": nav, "drawdown": nav / peak - 1.0})
    return pd.DataFrame(rows)


def _safe_mean(values: pd.Series) -> float:
    if values is None or len(values) == 0:
        return 0.0
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else 0.0


def _nullable_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _failure_reason(fill_count_ok: bool, quality_ok: bool, filled_rows: int) -> str:
    if filled_rows == 0:
        return "no matched buy fills for top candidates"
    reasons = []
    if not fill_count_ok:
        reasons.append("average filled boards per active day below threshold")
    if not quality_ok:
        reasons.append("filled return quality worse than adverse benchmark")
    return "; ".join(reasons)


def _write_outputs(out_dir: Path, summary: dict[str, object], daily: pd.DataFrame, fills: pd.DataFrame) -> None:
    (out_dir / "board_fillability_gate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    daily.to_csv(out_dir / "board_fillability_gate_daily.csv", index=False)
    fills.to_csv(out_dir / "board_fillability_gate_fills.csv", index=False)


if __name__ == "__main__":
    main()
