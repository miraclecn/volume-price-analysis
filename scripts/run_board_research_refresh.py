from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


DEFAULT_PREDICTIONS = "outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_predictions.csv"
DEFAULT_BOARD_DB = "outputs/limit_hit_research/board_execution.duckdb"
DEFAULT_MAIN_NAV = "outputs/ml/reports/profit_protect_continuous_wf_2020_2025_20260622/continuous_nav.csv"


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--board-db", default=DEFAULT_BOARD_DB)
    parser.add_argument("--main-nav", default=DEFAULT_MAIN_NAV)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    steps = build_refresh_steps(
        predictions=Path(args.predictions),
        board_db=Path(args.board_db),
        main_nav=Path(args.main_nav),
    )
    run_steps(steps, dry_run=args.dry_run)


def build_refresh_steps(*, predictions: Path, board_db: Path, main_nav: Path) -> list[Step]:
    python = sys.executable
    return [
        Step(
            "execution_data_audit",
            [
                python,
                "scripts/audit_board_execution_data.py",
                "--predictions",
                str(predictions),
                "--board-db",
                str(board_db),
                "--out-json",
                "outputs/limit_hit_research/reports/board_execution_data_audit.json",
            ],
        ),
        Step(
            "fillability_gate",
            [
                python,
                "scripts/research_board_fillability_gate.py",
                "--predictions",
                str(predictions),
                "--board-db",
                str(board_db),
                "--out-dir",
                "outputs/limit_hit_research/reports/board_fillability_gate_2020_2025",
            ],
        ),
        Step(
            "signal_diagnostics",
            [
                python,
                "scripts/research_board_signal_diagnostics.py",
                "--predictions",
                str(predictions),
                "--out-dir",
                "outputs/limit_hit_research/reports/board_signal_diagnostics_2020_2025",
                "--top-n",
                "5",
                "--min-segment-rows",
                "50",
            ],
        ),
        Step(
            "fill_proxy_sweep",
            [
                python,
                "scripts/research_board_fill_proxy_sweep.py",
                "--predictions",
                str(predictions),
                "--out-dir",
                "outputs/limit_hit_research/reports/board_fill_proxy_sweep_2020_2025",
            ],
        ),
        Step(
            "fill_aware_selection",
            [
                python,
                "scripts/research_board_fill_aware_selection.py",
                "--predictions",
                str(predictions),
                "--out-dir",
                "outputs/limit_hit_research/reports/board_fill_aware_selection_2020_2025",
            ],
        ),
        Step(
            "candidate_gate",
            [
                python,
                "scripts/research_board_candidate_gate.py",
                "--fillability-summary",
                "outputs/limit_hit_research/reports/board_fillability_gate_2020_2025/board_fillability_gate_summary.json",
                "--selection-metrics",
                "outputs/limit_hit_research/reports/board_fill_aware_selection_2020_2025/board_fill_aware_selection_metrics.csv",
                "--signal-profile",
                "outputs/limit_hit_research/reports/board_signal_diagnostics_2020_2025/board_signal_top_candidate_profile.csv",
                "--out-json",
                "outputs/limit_hit_research/reports/board_candidate_strategy_manifest.json",
                "--candidate-variant",
                "neutral_expected_pred_ret",
                "--min-live-fills-per-day",
                "2.0",
            ],
        ),
        Step(
            "overlay_profit_protect",
            [
                python,
                "scripts/research_board_overlay_profit_protect.py",
                "--main-nav",
                str(main_nav),
                "--out-dir",
                "outputs/limit_hit_research/reports/board_overlay_profit_protect_2020_2025",
            ],
        ),
        Step(
            "overlay_diagnostics",
            [
                python,
                "scripts/research_board_overlay_diagnostics.py",
                "--overlay-dir",
                "outputs/limit_hit_research/reports/board_overlay_profit_protect_2020_2025",
                "--variant",
                "board_neutral_expected_scale10",
            ],
        ),
        Step(
            "summary",
            [
                python,
                "scripts/summarize_board_research.py",
                "--candidate-manifest",
                "outputs/limit_hit_research/reports/board_candidate_strategy_manifest.json",
                "--overlay-metrics",
                "outputs/limit_hit_research/reports/board_overlay_profit_protect_2020_2025/board_overlay_metrics.csv",
                "--overlay-diagnostics",
                "outputs/limit_hit_research/reports/board_overlay_profit_protect_2020_2025/board_neutral_expected_scale10_diagnostics_summary.json",
                "--out-dir",
                "outputs/limit_hit_research/reports/board_research_summary",
            ],
        ),
    ]


def run_steps(steps: list[Step], *, dry_run: bool) -> None:
    for step in steps:
        print(f"[board-refresh] {step.name}: {' '.join(step.command)}", flush=True)
        if dry_run:
            continue
        subprocess.run(step.command, check=True)


if __name__ == "__main__":
    main()
