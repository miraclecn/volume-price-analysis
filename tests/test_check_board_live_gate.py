from __future__ import annotations

import json
from pathlib import Path

from scripts.check_board_live_gate import check_board_live_gate


def test_check_board_live_gate_fails_when_manifest_missing(tmp_path: Path) -> None:
    result = check_board_live_gate(tmp_path / "missing.json")

    assert result["ok"] is False
    assert result["status"] == "missing"


def test_check_board_live_gate_blocks_paper_only_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "paper_only",
                "live_sim_policy": "do_not_connect_until_status_is_live_candidate",
                "promotion_gate": {"blockers": ["missing fills"]},
            }
        ),
        encoding="utf-8",
    )

    result = check_board_live_gate(manifest)

    assert result["ok"] is False
    assert result["status"] == "paper_only"
    assert result["blockers"] == ["missing fills"]


def test_check_board_live_gate_allows_live_candidate_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "live_candidate",
                "candidate_variant": "neutral_expected_pred_ret",
                "candidate_rule": {"ranking": "pred_ret * fill_probability"},
            }
        ),
        encoding="utf-8",
    )

    result = check_board_live_gate(manifest)

    assert result["ok"] is True
    assert result["candidate_variant"] == "neutral_expected_pred_ret"
