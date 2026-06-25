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
DEFAULT_OUT_DIR = "outputs/limit_hit_research/reports/board_market_position_sweep_2020_2025"


@dataclass(frozen=True)
class MarketPositionVariant:
    name: str
    max_positions: int = 5
    name_weight: float = 0.01
    total_exposure_cap: float = 0.05
    min_pred_ret: float | None = None
    min_pred_win_prob: float | None = None
    market_rule: str = "none"
    prev_up_zero_below: float | None = None
    prev_up_half_below: float | None = None
    heat_min: float | None = None
    heat_max: float | None = None
    heat_half_outside: bool = False
    confidence_scale: bool = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_market_position_sweep(Path(args.predictions), Path(args.out_dir))


def run_market_position_sweep(predictions_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(predictions_path)
    variants = _variants()
    metrics_rows = []
    yearly_rows = []
    regime_rows = []
    for variant in variants:
        result = run_market_position_backtest(predictions, variant)
        result["nav"].to_csv(out_dir / f"{variant.name}_nav.csv", index=False)
        result["orders"].to_csv(out_dir / f"{variant.name}_orders.csv", index=False)
        metrics_rows.append(
            {
                "variant": variant.name,
                **asdict(variant),
                **result["metrics"],
                "buy_count": int(len(result["orders"])),
                "avg_exposure": float(result["daily"]["exposure"].mean()) if len(result["daily"]) else 0.0,
            }
        )
        for row in result["yearly_metrics"]:
            yearly_rows.append({"variant": variant.name, **row})
        regime_rows.extend(_regime_attribution(variant.name, result["orders"]))
    metrics = pd.DataFrame(metrics_rows).sort_values(["annual_return", "max_drawdown"], ascending=[False, False])
    yearly = pd.DataFrame(yearly_rows)
    regime = pd.DataFrame(regime_rows)
    metrics.to_csv(out_dir / "board_market_position_metrics.csv", index=False)
    yearly.to_csv(out_dir / "board_market_position_yearly_metrics.csv", index=False)
    regime.to_csv(out_dir / "board_market_position_regime_attribution.csv", index=False)
    manifest = {
        "predictions": str(predictions_path),
        "rows": int(len(predictions)),
        "variants": [asdict(v) for v in variants],
        "note": "Sealed-board overnight upper-bound sizing study. Not live executable without fillability gate.",
    }
    (out_dir / "board_market_position_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(metrics[["variant", "total_return", "annual_return", "max_drawdown", "sharpe", "calmar", "buy_count", "avg_exposure"]].to_string(index=False))
    print(f"wrote {out_dir}")


def run_market_position_backtest(
    predictions: pd.DataFrame,
    variant: MarketPositionVariant,
    *,
    initial_nav: float = 1_000_000.0,
) -> dict[str, pd.DataFrame | dict[str, float] | list[dict[str, float | int]]]:
    data = _normalize_predictions(predictions)
    nav = float(initial_nav)
    peak = nav
    nav_rows = []
    daily_rows = []
    order_rows = []
    for trade_date, day in data.groupby("trade_date", sort=True):
        candidates = _filter_candidates(day, variant)
        selected = candidates.sort_values(["pred_ret", "pred_win_prob", "code"], ascending=[False, False, True]).head(variant.max_positions)
        exposure = _market_exposure(day, selected, variant)
        day_ret = 0.0
        start_nav = nav
        if exposure > 0 and not selected.empty:
            total_weight = min(float(variant.total_exposure_cap), float(variant.name_weight) * len(selected)) * exposure
            weight = total_weight / len(selected)
            returns = pd.to_numeric(selected["target_ret_net"], errors="coerce").fillna(0.0)
            day_ret = float((returns * weight).sum())
            nav *= 1.0 + day_ret
            for row in selected.itertuples(index=False):
                order_rows.append(
                    {
                        "trade_date": trade_date,
                        "code": row.code,
                        "weight": weight,
                        "exposure": exposure,
                        "start_nav": start_nav,
                        "pnl": start_nav * weight * float(row.target_ret_net),
                        "target_ret_net": float(row.target_ret_net),
                        "target_win": int(row.target_win) if hasattr(row, "target_win") else int(row.target_ret_net > 0),
                        "second_board_success": bool(row.second_board_success) if hasattr(row, "second_board_success") else False,
                        "pred_ret": float(row.pred_ret),
                        "pred_win_prob": float(row.pred_win_prob),
                        "prev_up_ratio": float(row.prev_up_ratio),
                        "prev_sealed_count": float(row.prev_sealed_count),
                        "heat_bucket": _heat_bucket(float(row.prev_sealed_count)),
                        "prev_up_bucket": _prev_up_bucket(float(row.prev_up_ratio)),
                    }
                )
        peak = max(peak, nav)
        nav_rows.append({"trade_date": trade_date, "nav": nav, "drawdown": nav / peak - 1.0})
        daily_rows.append(
            {
                "trade_date": trade_date,
                "candidate_count": int(len(candidates)),
                "selected_count": int(len(selected)),
                "exposure": float(exposure),
                "day_return": day_ret,
                "prev_up_ratio": _day_value(day, "prev_up_ratio", 1.0),
                "prev_sealed_count": _day_value(day, "prev_sealed_count", 0.0),
                "avg_selected_pred_ret": float(selected["pred_ret"].mean()) if len(selected) else 0.0,
                "avg_selected_pred_win_prob": float(selected["pred_win_prob"].mean()) if len(selected) else 0.0,
            }
        )
    nav_frame = pd.DataFrame(nav_rows)
    orders = pd.DataFrame(order_rows)
    daily = pd.DataFrame(daily_rows)
    return {
        "nav": nav_frame,
        "orders": orders,
        "daily": daily,
        "metrics": _portfolio_metrics(nav_frame),
        "yearly_metrics": _yearly_metrics(nav_frame),
    }


def _normalize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "code", "target_ret_net", "pred_ret", "pred_win_prob", "prev_up_ratio", "prev_sealed_count"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {', '.join(missing)}")
    out = predictions.copy()
    out["trade_date"] = out["trade_date"].astype(str)
    out["code"] = out["code"].astype(str)
    for col in ["target_ret_net", "pred_ret", "pred_win_prob", "prev_up_ratio", "prev_sealed_count"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["target_win"] = pd.to_numeric(out.get("target_win", out["target_ret_net"] > 0), errors="coerce").fillna(0).astype(int)
    if "second_board_success" not in out.columns:
        out["second_board_success"] = False
    return out.dropna(subset=["target_ret_net", "pred_ret", "pred_win_prob", "prev_up_ratio", "prev_sealed_count"])


def _filter_candidates(day: pd.DataFrame, variant: MarketPositionVariant) -> pd.DataFrame:
    out = day.copy()
    if variant.min_pred_ret is not None:
        out = out[out["pred_ret"] >= float(variant.min_pred_ret)]
    if variant.min_pred_win_prob is not None:
        out = out[out["pred_win_prob"] >= float(variant.min_pred_win_prob)]
    return out


def _market_exposure(day: pd.DataFrame, selected: pd.DataFrame, variant: MarketPositionVariant) -> float:
    if day.empty:
        return 0.0
    exposure = 1.0
    prev_up = _day_value(day, "prev_up_ratio", 1.0)
    heat = _day_value(day, "prev_sealed_count", 0.0)
    if variant.market_rule in {"prev_up", "combo"}:
        if variant.prev_up_zero_below is not None and prev_up < variant.prev_up_zero_below:
            exposure = 0.0
        elif variant.prev_up_half_below is not None and prev_up < variant.prev_up_half_below:
            exposure *= 0.5
    if variant.market_rule in {"heat", "combo"}:
        inside_heat = True
        if variant.heat_min is not None and heat < variant.heat_min:
            inside_heat = False
        if variant.heat_max is not None and heat > variant.heat_max:
            inside_heat = False
        if not inside_heat:
            exposure *= 0.5 if variant.heat_half_outside else 0.0
    if variant.market_rule in {"heat_scale", "combo_heat_scale"}:
        if heat < 40:
            exposure *= 0.5
        elif heat >= 100:
            exposure *= 1.5
    if variant.market_rule in {"prev_up_scale", "combo_heat_scale"}:
        if prev_up < 0.4:
            exposure *= 0.75
        elif prev_up >= 0.8:
            exposure *= 1.25
    if variant.confidence_scale and not selected.empty:
        avg_pred = float(selected["pred_ret"].mean())
        if avg_pred < 0.015:
            exposure *= 0.5
        elif avg_pred >= 0.04:
            exposure *= 1.5
    return float(min(max(exposure, 0.0), 1.5))


def _regime_attribution(variant: str, orders: pd.DataFrame) -> list[dict[str, object]]:
    if orders.empty:
        return []
    rows = []
    for column in ["heat_bucket", "prev_up_bucket"]:
        for bucket, group in orders.groupby(column, sort=True):
            pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
            cost = pd.to_numeric(group["start_nav"], errors="coerce").fillna(0.0) * pd.to_numeric(group["weight"], errors="coerce").fillna(0.0)
            rows.append(
                {
                    "variant": variant,
                    "dimension": column,
                    "bucket": bucket,
                    "trades": int(len(group)),
                    "win_rate": float((pnl > 0).mean()),
                    "pnl_sum": float(pnl.sum()),
                    "return_on_cost_mean": float((pnl / cost.replace(0, np.nan)).mean()),
                    "target_ret_net_mean": float(pd.to_numeric(group["target_ret_net"], errors="coerce").mean()),
                    "pred_ret_mean": float(pd.to_numeric(group["pred_ret"], errors="coerce").mean()),
                }
            )
    return rows


def _variants() -> list[MarketPositionVariant]:
    variants = [
        MarketPositionVariant("base_top5_w01_total05"),
        MarketPositionVariant("base_top3_w01_total03", max_positions=3, total_exposure_cap=0.03),
        MarketPositionVariant("base_top5_w02_total10", name_weight=0.02, total_exposure_cap=0.10),
        MarketPositionVariant("pred_ret_ge_2pct_top5", min_pred_ret=0.02),
        MarketPositionVariant("pred_prob_ge_70_top5", min_pred_win_prob=0.70),
        MarketPositionVariant("confidence_scaled_top5", confidence_scale=True),
        MarketPositionVariant("heat_scaled_top5", market_rule="heat_scale"),
        MarketPositionVariant("prevup_scaled_top5", market_rule="prev_up_scale"),
        MarketPositionVariant("heat_prevup_scaled_top5", market_rule="combo_heat_scale"),
        MarketPositionVariant("heat_conf_scaled_top5", market_rule="heat_scale", confidence_scale=True),
    ]
    for heat_min, heat_max in [(20, 200), (30, 160), (40, 120), (60, 200)]:
        variants.append(
            MarketPositionVariant(
                name=f"heat_{heat_min}_{heat_max}_top5",
                market_rule="heat",
                heat_min=heat_min,
                heat_max=heat_max,
            )
        )
        variants.append(
            MarketPositionVariant(
                name=f"heat_{heat_min}_{heat_max}_half_outside_top5",
                market_rule="heat",
                heat_min=heat_min,
                heat_max=heat_max,
                heat_half_outside=True,
            )
        )
    for zero, half in [(0.30, 0.40), (0.35, 0.45), (0.40, 0.50)]:
        variants.append(
            MarketPositionVariant(
                name=f"prevup_{int(zero * 100)}_{int(half * 100)}_top5",
                market_rule="prev_up",
                prev_up_zero_below=zero,
                prev_up_half_below=half,
            )
        )
    for heat_min, heat_max, zero, half in [(20, 200, 0.30, 0.40), (30, 160, 0.35, 0.45), (40, 120, 0.35, 0.45)]:
        variants.append(
            MarketPositionVariant(
                name=f"combo_heat{heat_min}_{heat_max}_prevup{int(zero * 100)}_{int(half * 100)}",
                market_rule="combo",
                heat_min=heat_min,
                heat_max=heat_max,
                heat_half_outside=True,
                prev_up_zero_below=zero,
                prev_up_half_below=half,
            )
        )
        variants.append(
            MarketPositionVariant(
                name=f"combo_conf_heat{heat_min}_{heat_max}_prevup{int(zero * 100)}_{int(half * 100)}",
                market_rule="combo",
                heat_min=heat_min,
                heat_max=heat_max,
                heat_half_outside=True,
                prev_up_zero_below=zero,
                prev_up_half_below=half,
                confidence_scale=True,
            )
        )
    return variants


def _day_value(day: pd.DataFrame, column: str, default: float) -> float:
    values = pd.to_numeric(day[column], errors="coerce").dropna()
    return float(values.iloc[0]) if len(values) else default


def _heat_bucket(value: float) -> str:
    bins = [0, 20, 40, 60, 100, 200, float("inf")]
    labels = ["000_020", "020_040", "040_060", "060_100", "100_200", "200_plus"]
    for low, high, label in zip(bins[:-1], bins[1:], labels):
        if low <= value < high:
            return label
    return "unknown"


def _prev_up_bucket(value: float) -> str:
    bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01]
    labels = ["00_30", "30_40", "40_50", "50_60", "60_80", "80_100"]
    for low, high, label in zip(bins[:-1], bins[1:], labels):
        if low <= value < high:
            return label
    return "unknown"


if __name__ == "__main__":
    main()
