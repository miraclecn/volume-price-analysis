from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.summarize_board_research import build_board_research_summary, render_markdown


def test_board_research_summary_reports_paper_only_decision_and_overlay_lift(tmp_path: Path) -> None:
    manifest = tmp_path / "candidate.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "paper_only",
                "candidate_variant": "neutral_expected_pred_ret",
                "live_sim_policy": "do_not_connect_until_status_is_live_candidate",
                "candidate_rule": {
                    "ranking": "pred_ret * estimated_fill_probability",
                    "attempts_per_day": 5,
                    "single_name_weight": 0.01,
                    "max_attempted_sleeve": 0.05,
                },
                "candidate_metrics": {
                    "annual_return": 0.35,
                    "max_drawdown": -0.01,
                    "avg_expected_fills_per_day": 2.9,
                    "selected_return_mean": 0.048,
                },
                "promotion_gate": {
                    "fillability_ok": False,
                    "proxy_fill_ok": True,
                    "live_fill_ok": False,
                    "min_real_fills_per_active_day": 2.0,
                    "blockers": ["missing board execution database"],
                },
                "signal_profile": {
                    "all_sealed": {"target_ret_net_mean": 0.02, "win_rate": 0.65, "second_board_rate": 0.22},
                    "top_candidate": {
                        "target_ret_net_mean": 0.06,
                        "win_rate": 0.87,
                        "second_board_rate": 0.50,
                        "turnover_rate_mean": 3.7,
                        "adv20_amount_median": 83_000_000,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay.csv"
    pd.DataFrame(
        [
            {"variant": "main_only", "total_return": 1.0, "annual_return": 0.50, "max_drawdown": -0.18, "sharpe": 1.0, "calmar": 2.0},
            {"variant": "board_overlay", "total_return": 2.0, "annual_return": 0.80, "max_drawdown": -0.15, "sharpe": 2.0, "calmar": 4.0},
        ]
    ).to_csv(overlay, index=False)
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(
        json.dumps(
            {
                "main_board_corr": 0.12,
                "board_mean_return_on_main_down_days": 0.001,
                "both_down_rate": 0.04,
            }
        ),
        encoding="utf-8",
    )

    summary = build_board_research_summary(
        candidate_manifest_path=manifest,
        overlay_metrics_path=overlay,
        overlay_diagnostics_path=diagnostics,
    )
    markdown = render_markdown(summary)

    assert summary["status"] == "paper_only"
    assert summary["overlay"]["annual_return_lift"] == pytest.approx(0.30)
    assert "continue paper research" in summary["decision"]
    assert summary["overlay"]["diagnostics"]["main_board_corr"] == pytest.approx(0.12)
    assert "missing board execution database" in markdown
    assert "pred_ret * estimated_fill_probability" in markdown
    assert "Main/board daily correlation" in markdown
