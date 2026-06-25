from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_FILLABILITY_SUMMARY = "outputs/limit_hit_research/reports/board_fillability_gate_2020_2025/board_fillability_gate_summary.json"
DEFAULT_SELECTION_METRICS = "outputs/limit_hit_research/reports/board_fill_aware_selection_2020_2025/board_fill_aware_selection_metrics.csv"
DEFAULT_SIGNAL_PROFILE = "outputs/limit_hit_research/reports/board_signal_diagnostics_2020_2025/board_signal_top_candidate_profile.csv"
DEFAULT_OUT_JSON = "outputs/limit_hit_research/reports/board_candidate_strategy_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fillability-summary", default=DEFAULT_FILLABILITY_SUMMARY)
    parser.add_argument("--selection-metrics", default=DEFAULT_SELECTION_METRICS)
    parser.add_argument("--signal-profile", default=DEFAULT_SIGNAL_PROFILE)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--candidate-variant", default="neutral_expected_pred_ret")
    parser.add_argument("--min-live-fills-per-day", type=float, default=2.0)
    args = parser.parse_args()
    manifest = build_candidate_manifest(
        fillability_summary_path=Path(args.fillability_summary),
        selection_metrics_path=Path(args.selection_metrics),
        signal_profile_path=Path(args.signal_profile),
        candidate_variant=args.candidate_variant,
        min_live_fills_per_day=args.min_live_fills_per_day,
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def build_candidate_manifest(
    *,
    fillability_summary_path: Path,
    selection_metrics_path: Path,
    signal_profile_path: Path,
    candidate_variant: str,
    min_live_fills_per_day: float,
) -> dict[str, object]:
    fillability = _load_fillability(fillability_summary_path)
    selection = _load_variant(selection_metrics_path, candidate_variant)
    profile = _load_profile(signal_profile_path)
    live_fill_ok = bool(fillability.get("ok")) and float(fillability.get("avg_fills_per_active_day", 0.0)) >= float(min_live_fills_per_day)
    proxy_fill_ok = float(selection.get("avg_expected_fills_per_day", 0.0)) >= float(min_live_fills_per_day)
    annual_ok = float(selection.get("annual_return", 0.0)) > 0.0
    status = "live_candidate" if live_fill_ok and annual_ok else "paper_only"
    blockers = []
    if not bool(fillability.get("ok")):
        blockers.append(f"fillability gate failed: {fillability.get('reason', 'unknown')}")
    elif not live_fill_ok:
        blockers.append("real fills/day below live threshold")
    if not annual_ok:
        blockers.append("candidate annual_return is not positive")
    return {
        "status": status,
        "candidate_variant": candidate_variant,
        "candidate_rule": {
            "signal_universe": "sealed boards",
            "ranking": "pred_ret * estimated_fill_probability",
            "attempts_per_day": int(selection.get("max_candidates", 5)),
            "single_name_weight": float(selection.get("name_weight", 0.01)),
            "max_attempted_sleeve": float(selection.get("total_exposure_cap", 0.05)),
            "live_required_fill_probability_source": "broker/order fill logs",
        },
        "promotion_gate": {
            "min_real_fills_per_active_day": float(min_live_fills_per_day),
            "fillability_ok": bool(fillability.get("ok")),
            "live_fill_ok": bool(live_fill_ok),
            "proxy_fill_ok": bool(proxy_fill_ok),
            "annual_ok": bool(annual_ok),
            "blockers": blockers,
        },
        "candidate_metrics": _numeric_subset(
            selection,
            [
                "annual_return",
                "total_return",
                "max_drawdown",
                "sharpe",
                "avg_expected_fills_per_day",
                "avg_expected_exposure",
                "selected_return_mean",
                "selected_turnover_mean",
            ],
        ),
        "signal_profile": profile,
        "source_files": {
            "fillability_summary": str(fillability_summary_path),
            "selection_metrics": str(selection_metrics_path),
            "signal_profile": str(signal_profile_path),
        },
        "live_sim_policy": "do_not_connect_until_status_is_live_candidate",
    }


def _load_fillability(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"ok": False, "reason": f"missing fillability summary: {path}"}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_variant(path: Path, variant: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    metrics = pd.read_csv(path)
    rows = metrics[metrics["variant"].astype(str).eq(variant)]
    if rows.empty:
        raise ValueError(f"candidate variant not found: {variant}")
    return rows.iloc[0].to_dict()


def _load_profile(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    profile = pd.read_csv(path)
    out = {}
    for row in profile.to_dict(orient="records"):
        universe = str(row.pop("universe"))
        out[universe] = _coerce_record(row)
    return out


def _numeric_subset(row: dict[str, object], keys: list[str]) -> dict[str, float]:
    return {key: _to_float(row.get(key, 0.0)) for key in keys}


def _coerce_record(row: dict[str, object]) -> dict[str, object]:
    return {key: _to_float(value) if _is_number_like(value) else value for key, value in row.items()}


def _is_number_like(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _to_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
