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
DEFAULT_OUT_DIR = "outputs/limit_hit_research/reports/board_fill_proxy_sweep_2020_2025"


@dataclass(frozen=True)
class FillProxyVariant:
    name: str
    fill_probs: dict[str, float]
    max_candidates: int = 5
    name_weight: float = 0.01
    total_exposure_cap: float = 0.05
    extra_slippage_bps: float = 0.0
    min_pred_ret: float | None = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_fill_proxy_sweep(Path(args.predictions), Path(args.out_dir))


def run_fill_proxy_sweep(predictions_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = prepare_fill_proxy_frame(pd.read_csv(predictions_path))
    metrics_rows = []
    yearly_rows = []
    bucket_rows = []
    variants = _variants()
    for variant in variants:
        result = run_fill_proxy_backtest(predictions, variant)
        result["nav"].to_csv(out_dir / f"{variant.name}_nav.csv", index=False)
        result["orders"].to_csv(out_dir / f"{variant.name}_orders.csv", index=False)
        metrics_rows.append(
            {
                "variant": variant.name,
                **_variant_record(variant),
                **result["metrics"],
                "attempt_count": int(len(result["orders"])),
                "expected_fill_count": float(result["orders"]["fill_prob"].sum()) if len(result["orders"]) else 0.0,
                "avg_expected_fills_per_day": float(result["daily"]["expected_fill_count"].mean()) if len(result["daily"]) else 0.0,
                "avg_expected_exposure": float(result["daily"]["expected_exposure"].mean()) if len(result["daily"]) else 0.0,
            }
        )
        for row in result["yearly_metrics"]:
            yearly_rows.append({"variant": variant.name, **row})
        bucket_rows.extend(_bucket_attribution(variant.name, result["orders"]))
    metrics = pd.DataFrame(metrics_rows).sort_values(["annual_return", "max_drawdown"], ascending=[False, False])
    yearly = pd.DataFrame(yearly_rows)
    buckets = pd.DataFrame(bucket_rows)
    metrics.to_csv(out_dir / "board_fill_proxy_metrics.csv", index=False)
    yearly.to_csv(out_dir / "board_fill_proxy_yearly_metrics.csv", index=False)
    buckets.to_csv(out_dir / "board_fill_proxy_bucket_attribution.csv", index=False)
    manifest = {
        "predictions": str(predictions_path),
        "rows": int(len(predictions)),
        "variants": [_variant_record(v) for v in variants],
        "note": "Daily-bar fillability proxy by turnover bucket. This is not a substitute for broker fills.",
    }
    (out_dir / "board_fill_proxy_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(metrics[["variant", "total_return", "annual_return", "max_drawdown", "sharpe", "attempt_count", "expected_fill_count", "avg_expected_fills_per_day"]].to_string(index=False))
    print(f"wrote {out_dir}")


def prepare_fill_proxy_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "code", "target_ret_net", "pred_ret", "pred_win_prob", "turnover_rate"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {', '.join(missing)}")
    out = predictions.copy()
    out["trade_date"] = out["trade_date"].astype(str)
    out["code"] = out["code"].astype(str)
    for col in ["target_ret_net", "pred_ret", "pred_win_prob", "turnover_rate"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["target_ret_net", "pred_ret", "pred_win_prob", "turnover_rate"])
    out["turnover_bucket"] = _turnover_bucket(out["turnover_rate"])
    return out


def run_fill_proxy_backtest(
    predictions: pd.DataFrame,
    variant: FillProxyVariant,
    *,
    initial_nav: float = 1_000_000.0,
) -> dict[str, pd.DataFrame | dict[str, float] | list[dict[str, float | int]]]:
    nav = float(initial_nav)
    peak = nav
    nav_rows = []
    daily_rows = []
    order_rows = []
    extra_cost = float(variant.extra_slippage_bps) / 10000.0
    for trade_date, day in predictions.groupby("trade_date", sort=True):
        candidates = day.copy()
        if variant.min_pred_ret is not None:
            candidates = candidates[pd.to_numeric(candidates["pred_ret"], errors="coerce") >= float(variant.min_pred_ret)]
        selected = candidates.sort_values(["pred_ret", "pred_win_prob", "code"], ascending=[False, False, True]).head(variant.max_candidates).copy()
        expected_day_return = 0.0
        expected_fill_count = 0.0
        expected_exposure = 0.0
        if not selected.empty:
            base_weight = min(float(variant.total_exposure_cap), float(variant.name_weight) * len(selected)) / len(selected)
            start_nav = nav
            for row in selected.itertuples(index=False):
                fill_prob = float(variant.fill_probs.get(str(row.turnover_bucket), 0.0))
                adjusted_ret = float(row.target_ret_net) - extra_cost
                contribution = base_weight * fill_prob * adjusted_ret
                expected_day_return += contribution
                expected_fill_count += fill_prob
                expected_exposure += base_weight * fill_prob
                order_rows.append(
                    {
                        "trade_date": trade_date,
                        "code": row.code,
                        "weight_if_filled": base_weight,
                        "fill_prob": fill_prob,
                        "expected_weight": base_weight * fill_prob,
                        "start_nav": start_nav,
                        "expected_pnl": start_nav * contribution,
                        "target_ret_net": float(row.target_ret_net),
                        "adjusted_ret_net": adjusted_ret,
                        "pred_ret": float(row.pred_ret),
                        "pred_win_prob": float(row.pred_win_prob),
                        "turnover_rate": float(row.turnover_rate),
                        "turnover_bucket": row.turnover_bucket,
                    }
                )
            nav *= 1.0 + expected_day_return
        peak = max(peak, nav)
        nav_rows.append({"trade_date": trade_date, "nav": nav, "drawdown": nav / peak - 1.0})
        daily_rows.append(
            {
                "trade_date": trade_date,
                "attempt_count": int(len(selected)),
                "expected_fill_count": expected_fill_count,
                "expected_exposure": expected_exposure,
                "expected_day_return": expected_day_return,
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


def _bucket_attribution(variant: str, orders: pd.DataFrame) -> list[dict[str, object]]:
    if orders.empty:
        return []
    rows = []
    for bucket, group in orders.groupby("turnover_bucket", sort=True):
        expected_pnl = pd.to_numeric(group["expected_pnl"], errors="coerce").fillna(0.0)
        expected_weight = pd.to_numeric(group["expected_weight"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "variant": variant,
                "turnover_bucket": bucket,
                "attempts": int(len(group)),
                "expected_fills": float(pd.to_numeric(group["fill_prob"], errors="coerce").sum()),
                "expected_pnl_sum": float(expected_pnl.sum()),
                "expected_weight_sum": float(expected_weight.sum()),
                "target_ret_net_mean": float(pd.to_numeric(group["target_ret_net"], errors="coerce").mean()),
                "adjusted_ret_net_mean": float(pd.to_numeric(group["adjusted_ret_net"], errors="coerce").mean()),
                "pred_ret_mean": float(pd.to_numeric(group["pred_ret"], errors="coerce").mean()),
                "turnover_rate_mean": float(pd.to_numeric(group["turnover_rate"], errors="coerce").mean()),
            }
        )
    return rows


def _variants() -> list[FillProxyVariant]:
    optimistic = {"turn_q1": 0.50, "turn_q2": 0.70, "turn_q3": 0.80, "turn_q4": 0.90, "turn_q5": 0.95}
    neutral = {"turn_q1": 0.25, "turn_q2": 0.45, "turn_q3": 0.60, "turn_q4": 0.75, "turn_q5": 0.85}
    conservative = {"turn_q1": 0.10, "turn_q2": 0.25, "turn_q3": 0.40, "turn_q4": 0.55, "turn_q5": 0.70}
    severe_adverse = {"turn_q1": 0.05, "turn_q2": 0.15, "turn_q3": 0.40, "turn_q4": 0.70, "turn_q5": 0.90}
    return [
        FillProxyVariant("optimistic_turnover_fill", optimistic),
        FillProxyVariant("neutral_turnover_fill", neutral),
        FillProxyVariant("conservative_turnover_fill", conservative),
        FillProxyVariant("severe_adverse_turnover_fill", severe_adverse),
        FillProxyVariant("neutral_turnover_fill_extra50bps", neutral, extra_slippage_bps=50.0),
        FillProxyVariant("conservative_turnover_fill_extra50bps", conservative, extra_slippage_bps=50.0),
        FillProxyVariant("severe_adverse_turnover_fill_extra50bps", severe_adverse, extra_slippage_bps=50.0),
        FillProxyVariant("neutral_pred_ret_ge2pct", neutral, min_pred_ret=0.02),
    ]


def _turnover_bucket(values: pd.Series) -> pd.Series:
    labels = [f"turn_q{i}" for i in range(1, 6)]
    return pd.qcut(pd.to_numeric(values, errors="coerce").rank(method="first"), 5, labels=labels).astype(str)


def _variant_record(variant: FillProxyVariant) -> dict[str, object]:
    record = asdict(variant)
    record["fill_probs"] = json.dumps(variant.fill_probs, sort_keys=True)
    return record


if __name__ == "__main__":
    main()
