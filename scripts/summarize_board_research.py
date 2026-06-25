from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_CANDIDATE_MANIFEST = "outputs/limit_hit_research/reports/board_candidate_strategy_manifest.json"
DEFAULT_OVERLAY_METRICS = "outputs/limit_hit_research/reports/board_overlay_profit_protect_2020_2025/board_overlay_metrics.csv"
DEFAULT_OVERLAY_DIAGNOSTICS = "outputs/limit_hit_research/reports/board_overlay_profit_protect_2020_2025/board_neutral_expected_scale10_diagnostics_summary.json"
DEFAULT_OUT_DIR = "outputs/limit_hit_research/reports/board_research_summary"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--overlay-metrics", default=DEFAULT_OVERLAY_METRICS)
    parser.add_argument("--overlay-diagnostics", default=DEFAULT_OVERLAY_DIAGNOSTICS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = build_board_research_summary(
        candidate_manifest_path=Path(args.candidate_manifest),
        overlay_metrics_path=Path(args.overlay_metrics),
        overlay_diagnostics_path=Path(args.overlay_diagnostics),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "board_research_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "board_research_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))


def build_board_research_summary(
    *,
    candidate_manifest_path: Path,
    overlay_metrics_path: Path,
    overlay_diagnostics_path: Path | None = None,
) -> dict[str, object]:
    candidate = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    overlay = pd.read_csv(overlay_metrics_path)
    main = _variant_row(overlay, "main_only")
    best_overlay = overlay.sort_values(["annual_return", "max_drawdown"], ascending=[False, False]).iloc[0].to_dict()
    diagnostics = _load_overlay_diagnostics(overlay_diagnostics_path)
    candidate_metrics = candidate.get("candidate_metrics", {})
    profile = candidate.get("signal_profile", {})
    top_profile = profile.get("top_candidate", {})
    all_profile = profile.get("all_sealed", {})
    blockers = candidate.get("promotion_gate", {}).get("blockers", [])
    return {
        "status": candidate.get("status", "unknown"),
        "live_sim_policy": candidate.get("live_sim_policy", ""),
        "candidate_variant": candidate.get("candidate_variant", ""),
        "candidate_rule": candidate.get("candidate_rule", {}),
        "candidate_metrics": candidate_metrics,
        "promotion_gate": candidate.get("promotion_gate", {}),
        "blockers": blockers,
        "signal_law": {
            "all_sealed_mean_return": _float(all_profile.get("target_ret_net_mean")),
            "all_sealed_win_rate": _float(all_profile.get("win_rate")),
            "all_sealed_second_board_rate": _float(all_profile.get("second_board_rate")),
            "top_candidate_mean_return": _float(top_profile.get("target_ret_net_mean")),
            "top_candidate_win_rate": _float(top_profile.get("win_rate")),
            "top_candidate_second_board_rate": _float(top_profile.get("second_board_rate")),
            "top_candidate_turnover_mean": _float(top_profile.get("turnover_rate_mean")),
            "top_candidate_adv20_median": _float(top_profile.get("adv20_amount_median")),
        },
        "overlay": {
            "main_only": _metric_record(main),
            "best_overlay_variant": str(best_overlay.get("variant")),
            "best_overlay": _metric_record(best_overlay),
            "annual_return_lift": _float(best_overlay.get("annual_return")) - _float(main.get("annual_return")),
            "max_drawdown_change": _float(best_overlay.get("max_drawdown")) - _float(main.get("max_drawdown")),
            "diagnostics": diagnostics,
        },
        "decision": _decision(candidate),
    }


def render_markdown(summary: dict[str, object]) -> str:
    candidate_rule = summary["candidate_rule"]
    candidate_metrics = summary["candidate_metrics"]
    gate = summary["promotion_gate"]
    signal = summary["signal_law"]
    overlay = summary["overlay"]
    blockers = summary["blockers"] or ["none"]
    return "\n".join(
        [
            "# Board Research Decision Summary",
            "",
            f"Status: `{summary['status']}`",
            f"Decision: {summary['decision']}",
            "",
            "## Candidate Rule",
            "",
            f"- Variant: `{summary['candidate_variant']}`",
            f"- Ranking: `{candidate_rule.get('ranking')}`",
            f"- Attempts/day: `{candidate_rule.get('attempts_per_day')}`",
            f"- Single-name weight: `{_pct(candidate_rule.get('single_name_weight'))}`",
            f"- Max attempted sleeve: `{_pct(candidate_rule.get('max_attempted_sleeve'))}`",
            "",
            "## Candidate Proxy Metrics",
            "",
            f"- Annual return: `{_pct(candidate_metrics.get('annual_return'))}`",
            f"- Max drawdown: `{_pct(candidate_metrics.get('max_drawdown'))}`",
            f"- Expected fills/day: `{_num(candidate_metrics.get('avg_expected_fills_per_day'))}`",
            f"- Selected mean return: `{_pct(candidate_metrics.get('selected_return_mean'))}`",
            "",
            "## Signal Law",
            "",
            f"- All sealed boards: mean `{_pct(signal.get('all_sealed_mean_return'))}`, win `{_pct(signal.get('all_sealed_win_rate'))}`, second-board `{_pct(signal.get('all_sealed_second_board_rate'))}`",
            f"- Top candidates: mean `{_pct(signal.get('top_candidate_mean_return'))}`, win `{_pct(signal.get('top_candidate_win_rate'))}`, second-board `{_pct(signal.get('top_candidate_second_board_rate'))}`",
            f"- Top candidate turnover mean: `{_num(signal.get('top_candidate_turnover_mean'))}`",
            f"- Top candidate median ADV20 amount: `{_money(signal.get('top_candidate_adv20_median'))}`",
            "",
            "## Profit-Protect Overlay",
            "",
            f"- Main only annual/maxDD: `{_pct(overlay['main_only']['annual_return'])}` / `{_pct(overlay['main_only']['max_drawdown'])}`",
            f"- Best overlay: `{overlay['best_overlay_variant']}`",
            f"- Best overlay annual/maxDD: `{_pct(overlay['best_overlay']['annual_return'])}` / `{_pct(overlay['best_overlay']['max_drawdown'])}`",
            f"- Annual lift: `{_pct(overlay['annual_return_lift'])}`",
            f"- MaxDD change: `{_pct(overlay['max_drawdown_change'])}`",
            f"- Main/board daily correlation: `{_num(overlay['diagnostics'].get('main_board_corr'))}`",
            f"- Board return on main down days: `{_pct(overlay['diagnostics'].get('board_mean_return_on_main_down_days'))}`",
            f"- Both-down day rate: `{_pct(overlay['diagnostics'].get('both_down_rate'))}`",
            "",
            "## Promotion Gate",
            "",
            f"- Fillability ok: `{gate.get('fillability_ok')}`",
            f"- Proxy fill ok: `{gate.get('proxy_fill_ok')}`",
            f"- Live fill ok: `{gate.get('live_fill_ok')}`",
            f"- Min real fills/day: `{_num(gate.get('min_real_fills_per_active_day'))}`",
            "- Blockers:",
            *[f"  - {blocker}" for blocker in blockers],
            "",
            "Live policy: do not connect to live sim until status is `live_candidate`.",
            "",
        ]
    )


def _decision(candidate: dict[str, object]) -> str:
    status = candidate.get("status")
    if status == "live_candidate":
        return "eligible for controlled live-sim integration after separate implementation review"
    return "continue paper research and collect/import real board execution fills"


def _variant_row(frame: pd.DataFrame, variant: str) -> dict[str, object]:
    rows = frame[frame["variant"].astype(str).eq(variant)]
    if rows.empty:
        raise ValueError(f"missing overlay variant: {variant}")
    return rows.iloc[0].to_dict()


def _load_overlay_diagnostics(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {key: _float(value) for key, value in raw.items()}


def _metric_record(row: dict[str, object]) -> dict[str, float]:
    return {
        "total_return": _float(row.get("total_return")),
        "annual_return": _float(row.get("annual_return")),
        "max_drawdown": _float(row.get("max_drawdown")),
        "sharpe": _float(row.get("sharpe")),
        "calmar": _float(row.get("calmar")),
    }


def _float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pct(value: object) -> str:
    return f"{_float(value) * 100:.2f}%"


def _num(value: object) -> str:
    return f"{_float(value):.2f}"


def _money(value: object) -> str:
    return f"{_float(value):,.0f}"


if __name__ == "__main__":
    main()
