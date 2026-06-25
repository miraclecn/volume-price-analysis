from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.research_board_candidate_gate import build_candidate_manifest


def test_candidate_manifest_remains_paper_only_when_fillability_gate_fails(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, fillability_ok=False)

    manifest = build_candidate_manifest(
        fillability_summary_path=paths["fillability"],
        selection_metrics_path=paths["metrics"],
        signal_profile_path=paths["profile"],
        candidate_variant="neutral_expected_pred_ret",
        min_live_fills_per_day=2.0,
    )

    assert manifest["status"] == "paper_only"
    assert manifest["promotion_gate"]["proxy_fill_ok"] is True
    assert manifest["promotion_gate"]["live_fill_ok"] is False
    assert "fillability gate failed" in manifest["promotion_gate"]["blockers"][0]
    assert manifest["live_sim_policy"] == "do_not_connect_until_status_is_live_candidate"


def test_candidate_manifest_can_be_live_candidate_when_real_fill_gate_passes(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, fillability_ok=True)

    manifest = build_candidate_manifest(
        fillability_summary_path=paths["fillability"],
        selection_metrics_path=paths["metrics"],
        signal_profile_path=paths["profile"],
        candidate_variant="neutral_expected_pred_ret",
        min_live_fills_per_day=2.0,
    )

    assert manifest["status"] == "live_candidate"
    assert manifest["promotion_gate"]["blockers"] == []


def test_candidate_manifest_rejects_missing_variant(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, fillability_ok=True)

    with pytest.raises(ValueError, match="candidate variant not found"):
        build_candidate_manifest(
            fillability_summary_path=paths["fillability"],
            selection_metrics_path=paths["metrics"],
            signal_profile_path=paths["profile"],
            candidate_variant="missing",
            min_live_fills_per_day=2.0,
        )


def _write_inputs(tmp_path: Path, *, fillability_ok: bool) -> dict[str, Path]:
    fillability = tmp_path / "fillability.json"
    fillability.write_text(
        json.dumps(
            {
                "ok": fillability_ok,
                "reason": "pass" if fillability_ok else "missing board execution database",
                "avg_fills_per_active_day": 2.4 if fillability_ok else 0.0,
            }
        ),
        encoding="utf-8",
    )
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame(
        [
            {
                "variant": "neutral_expected_pred_ret",
                "annual_return": 0.35,
                "total_return": 4.3,
                "max_drawdown": -0.002,
                "sharpe": 19.9,
                "avg_expected_fills_per_day": 2.9,
                "avg_expected_exposure": 0.029,
                "selected_return_mean": 0.048,
                "selected_turnover_mean": 9.4,
                "max_candidates": 5,
                "name_weight": 0.01,
                "total_exposure_cap": 0.05,
            }
        ]
    ).to_csv(metrics, index=False)
    profile = tmp_path / "profile.csv"
    pd.DataFrame(
        [
            {"universe": "all_sealed", "rows": 100, "target_ret_net_mean": 0.02},
            {"universe": "top_candidate", "rows": 10, "target_ret_net_mean": 0.06},
        ]
    ).to_csv(profile, index=False)
    return {"fillability": fillability, "metrics": metrics, "profile": profile}
