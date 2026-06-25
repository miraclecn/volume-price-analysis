from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.research_board_overnight_model import _portfolio_metrics, _yearly_metrics


DEFAULT_PREDICTIONS = "outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_predictions.csv"
DEFAULT_OUT_DIR = "outputs/limit_hit_research/reports/board_fill_stress_2020_2025"


@dataclass(frozen=True)
class FillStressVariant:
    name: str
    max_candidates: int = 5
    max_fills: int = 5
    name_weight: float = 0.01
    total_exposure_cap: float = 0.05
    fill_mode: str = "top_pred"
    extra_slippage_bps: float = 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_fill_stress(Path(args.predictions), Path(args.out_dir))


def run_fill_stress(predictions_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(predictions_path)
    variants = _variants()
    metrics_rows = []
    yearly_rows = []
    for variant in variants:
        result = run_fill_stress_backtest(predictions, variant)
        result["nav"].to_csv(out_dir / f"{variant.name}_nav.csv", index=False)
        result["orders"].to_csv(out_dir / f"{variant.name}_orders.csv", index=False)
        metrics_rows.append({"variant": variant.name, **asdict(variant), **result["metrics"], "fill_count": int(len(result["orders"]))})
        for row in result["yearly_metrics"]:
            yearly_rows.append({"variant": variant.name, **row})
    metrics = pd.DataFrame(metrics_rows).sort_values(["annual_return", "max_drawdown"], ascending=[False, False])
    yearly = pd.DataFrame(yearly_rows)
    metrics.to_csv(out_dir / "board_fill_stress_metrics.csv", index=False)
    yearly.to_csv(out_dir / "board_fill_stress_yearly_metrics.csv", index=False)
    manifest = {
        "predictions": str(predictions_path),
        "rows": int(len(predictions)),
        "variants": [asdict(v) for v in variants],
    }
    (out_dir / "board_fill_stress_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(metrics[["variant", "total_return", "annual_return", "max_drawdown", "sharpe", "calmar", "fill_count"]].to_string(index=False))
    print(f"wrote {out_dir}")


def run_fill_stress_backtest(predictions: pd.DataFrame, variant: FillStressVariant, *, initial_nav: float = 1_000_000.0) -> dict[str, pd.DataFrame | dict[str, float] | list[dict[str, float | int]]]:
    nav = float(initial_nav)
    peak = nav
    nav_rows = []
    order_rows = []
    extra_cost = float(variant.extra_slippage_bps) / 10000.0
    for date, day in predictions.groupby("trade_date", sort=True):
        candidates = day.sort_values(["pred_ret", "pred_win_prob", "code"], ascending=[False, False, True]).head(variant.max_candidates).copy()
        filled = _select_fills(candidates, variant, date)
        if not filled.empty:
            total_weight = min(float(variant.total_exposure_cap), float(variant.name_weight) * len(filled))
            weight = total_weight / len(filled)
            adjusted_returns = pd.to_numeric(filled["target_ret_net"], errors="coerce").fillna(0.0) - extra_cost
            day_ret = float((adjusted_returns * weight).sum())
            start_nav = nav
            nav *= 1.0 + day_ret
            for row, adjusted_ret in zip(filled.itertuples(index=False), adjusted_returns):
                order_rows.append(
                    {
                        "trade_date": date,
                        "code": row.code,
                        "weight": weight,
                        "start_nav": start_nav,
                        "pnl": start_nav * weight * float(adjusted_ret),
                        "target_ret_net": float(row.target_ret_net),
                        "adjusted_ret_net": float(adjusted_ret),
                        "pred_ret": float(row.pred_ret),
                        "pred_win_prob": float(row.pred_win_prob),
                        "fill_mode": variant.fill_mode,
                    }
                )
        peak = max(peak, nav)
        nav_rows.append({"trade_date": date, "nav": nav, "drawdown": nav / peak - 1.0})
    nav_frame = pd.DataFrame(nav_rows)
    orders = pd.DataFrame(order_rows)
    return {"nav": nav_frame, "orders": orders, "metrics": _portfolio_metrics(nav_frame), "yearly_metrics": _yearly_metrics(nav_frame)}


def _select_fills(candidates: pd.DataFrame, variant: FillStressVariant, date: object) -> pd.DataFrame:
    if candidates.empty or variant.max_fills <= 0:
        return candidates.head(0)
    n = min(int(variant.max_fills), len(candidates))
    if variant.fill_mode == "top_pred":
        return candidates.head(n)
    if variant.fill_mode == "random":
        return candidates.sample(n=n, random_state=int(str(date).replace("-", "")))
    if variant.fill_mode == "adverse_realized":
        return candidates.sort_values(["target_ret_net", "pred_ret", "code"], ascending=[True, False, True]).head(n)
    if variant.fill_mode == "best_realized":
        return candidates.sort_values(["target_ret_net", "pred_ret", "code"], ascending=[False, False, True]).head(n)
    raise ValueError(f"unknown fill_mode: {variant.fill_mode}")


def _variants() -> list[FillStressVariant]:
    variants = []
    for fills in [5, 3, 2, 1]:
        variants.append(FillStressVariant(name=f"ideal_top5_fill{fills}", max_fills=fills, fill_mode="top_pred"))
        variants.append(FillStressVariant(name=f"random_top5_fill{fills}", max_fills=fills, fill_mode="random"))
        variants.append(FillStressVariant(name=f"adverse_top5_fill{fills}", max_fills=fills, fill_mode="adverse_realized"))
    for extra_slip in [10.0, 30.0, 50.0, 100.0]:
        variants.append(FillStressVariant(name=f"ideal_top5_fill5_extra{int(extra_slip)}bps", max_fills=5, fill_mode="top_pred", extra_slippage_bps=extra_slip))
        variants.append(FillStressVariant(name=f"adverse_top5_fill2_extra{int(extra_slip)}bps", max_fills=2, fill_mode="adverse_realized", extra_slippage_bps=extra_slip))
    return variants


if __name__ == "__main__":
    main()
