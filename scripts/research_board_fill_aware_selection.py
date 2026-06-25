from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.research_board_fill_proxy_sweep import prepare_fill_proxy_frame
from scripts.research_board_overnight_model import _portfolio_metrics, _yearly_metrics


DEFAULT_PREDICTIONS = "outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_predictions.csv"
DEFAULT_OUT_DIR = "outputs/limit_hit_research/reports/board_fill_aware_selection_2020_2025"


@dataclass(frozen=True)
class FillAwareVariant:
    name: str
    fill_probs: dict[str, float]
    selection_policy: str
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
    run_fill_aware_selection_sweep(Path(args.predictions), Path(args.out_dir))


def run_fill_aware_selection_sweep(predictions_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = prepare_fill_proxy_frame(pd.read_csv(predictions_path))
    variants = _variants()
    metrics_rows = []
    yearly_rows = []
    policy_rows = []
    for variant in variants:
        result = run_fill_aware_backtest(predictions, variant)
        result["nav"].to_csv(out_dir / f"{variant.name}_nav.csv", index=False)
        result["orders"].to_csv(out_dir / f"{variant.name}_orders.csv", index=False)
        orders = result["orders"]
        metrics_rows.append(
            {
                "variant": variant.name,
                **_variant_record(variant),
                **result["metrics"],
                "attempt_count": int(len(orders)),
                "expected_fill_count": float(orders["fill_prob"].sum()) if len(orders) else 0.0,
                "avg_expected_fills_per_day": float(result["daily"]["expected_fill_count"].mean()) if len(result["daily"]) else 0.0,
                "avg_expected_exposure": float(result["daily"]["expected_exposure"].mean()) if len(result["daily"]) else 0.0,
                "selected_return_mean": float(pd.to_numeric(orders["target_ret_net"], errors="coerce").mean()) if len(orders) else 0.0,
                "selected_turnover_mean": float(pd.to_numeric(orders["turnover_rate"], errors="coerce").mean()) if len(orders) else 0.0,
            }
        )
        for row in result["yearly_metrics"]:
            yearly_rows.append({"variant": variant.name, **row})
        policy_rows.extend(_policy_attribution(variant.name, orders))
    metrics = pd.DataFrame(metrics_rows).sort_values(["annual_return", "max_drawdown"], ascending=[False, False])
    yearly = pd.DataFrame(yearly_rows)
    policy = pd.DataFrame(policy_rows)
    metrics.to_csv(out_dir / "board_fill_aware_selection_metrics.csv", index=False)
    yearly.to_csv(out_dir / "board_fill_aware_selection_yearly_metrics.csv", index=False)
    policy.to_csv(out_dir / "board_fill_aware_selection_bucket_attribution.csv", index=False)
    manifest = {
        "predictions": str(predictions_path),
        "rows": int(len(predictions)),
        "variants": [_variant_record(v) for v in variants],
        "note": "Compares alpha-first vs fill-aware sealed-board selection under turnover-based fill proxies.",
    }
    (out_dir / "board_fill_aware_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(metrics[["variant", "annual_return", "max_drawdown", "attempt_count", "expected_fill_count", "avg_expected_fills_per_day", "selected_return_mean", "selected_turnover_mean"]].to_string(index=False))
    print(f"wrote {out_dir}")


def run_fill_aware_backtest(
    predictions: pd.DataFrame,
    variant: FillAwareVariant,
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
        selected = select_candidates(candidates, variant)
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
                        "selection_policy": variant.selection_policy,
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


def select_candidates(candidates: pd.DataFrame, variant: FillAwareVariant) -> pd.DataFrame:
    if candidates.empty:
        return candidates.head(0)
    frame = candidates.copy()
    frame["fill_prob"] = frame["turnover_bucket"].map(variant.fill_probs).fillna(0.0).astype(float)
    frame["expected_pred_ret"] = pd.to_numeric(frame["pred_ret"], errors="coerce").fillna(0.0) * frame["fill_prob"]
    frame["turnover_bucket_rank"] = frame["turnover_bucket"].map({"turn_q1": 1, "turn_q2": 2, "turn_q3": 3, "turn_q4": 4, "turn_q5": 5}).fillna(9).astype(int)
    policy = variant.selection_policy
    if policy == "alpha_top":
        sort_cols = ["pred_ret", "pred_win_prob", "code"]
        ascending = [False, False, True]
    elif policy == "fill_prob_top":
        sort_cols = ["fill_prob", "pred_ret", "pred_win_prob", "code"]
        ascending = [False, False, False, True]
    elif policy == "expected_pred_ret":
        sort_cols = ["expected_pred_ret", "pred_ret", "pred_win_prob", "code"]
        ascending = [False, False, False, True]
    elif policy == "low_turnover_first":
        sort_cols = ["turnover_bucket_rank", "pred_ret", "pred_win_prob", "code"]
        ascending = [True, False, False, True]
    elif policy == "alpha_low_turnover_only":
        frame = frame[frame["turnover_bucket"].isin(["turn_q1", "turn_q2", "turn_q3"])]
        sort_cols = ["pred_ret", "pred_win_prob", "code"]
        ascending = [False, False, True]
    else:
        raise ValueError(f"unknown selection_policy: {policy}")
    return frame.sort_values(sort_cols, ascending=ascending).head(variant.max_candidates).reset_index(drop=True)


def _policy_attribution(variant: str, orders: pd.DataFrame) -> list[dict[str, object]]:
    if orders.empty:
        return []
    rows = []
    for bucket, group in orders.groupby("turnover_bucket", sort=True):
        rows.append(
            {
                "variant": variant,
                "turnover_bucket": bucket,
                "attempts": int(len(group)),
                "expected_fills": float(pd.to_numeric(group["fill_prob"], errors="coerce").sum()),
                "expected_pnl_sum": float(pd.to_numeric(group["expected_pnl"], errors="coerce").sum()),
                "target_ret_net_mean": float(pd.to_numeric(group["target_ret_net"], errors="coerce").mean()),
                "pred_ret_mean": float(pd.to_numeric(group["pred_ret"], errors="coerce").mean()),
                "turnover_rate_mean": float(pd.to_numeric(group["turnover_rate"], errors="coerce").mean()),
            }
        )
    return rows


def _variants() -> list[FillAwareVariant]:
    neutral = {"turn_q1": 0.25, "turn_q2": 0.45, "turn_q3": 0.60, "turn_q4": 0.75, "turn_q5": 0.85}
    conservative = {"turn_q1": 0.10, "turn_q2": 0.25, "turn_q3": 0.40, "turn_q4": 0.55, "turn_q5": 0.70}
    severe_adverse = {"turn_q1": 0.05, "turn_q2": 0.15, "turn_q3": 0.40, "turn_q4": 0.70, "turn_q5": 0.90}
    variants = []
    for name, probs in [
        ("neutral", neutral),
        ("conservative", conservative),
        ("severe", severe_adverse),
    ]:
        for policy in [
            "alpha_top",
            "fill_prob_top",
            "expected_pred_ret",
            "low_turnover_first",
            "alpha_low_turnover_only",
        ]:
            variants.append(FillAwareVariant(f"{name}_{policy}", probs, policy))
        variants.append(FillAwareVariant(f"{name}_expected_pred_ret_extra50bps", probs, "expected_pred_ret", extra_slippage_bps=50.0))
    return variants


def _variant_record(variant: FillAwareVariant) -> dict[str, object]:
    record = asdict(variant)
    record["fill_probs"] = json.dumps(variant.fill_probs, sort_keys=True)
    return record


if __name__ == "__main__":
    main()
