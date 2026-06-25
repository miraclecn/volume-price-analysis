from __future__ import annotations

from pathlib import Path

from scripts.run_board_research_refresh import build_refresh_steps, run_steps


def test_build_refresh_steps_orders_downstream_research_pipeline() -> None:
    steps = build_refresh_steps(
        predictions=Path("pred.csv"),
        board_db=Path("board.duckdb"),
        main_nav=Path("main_nav.csv"),
    )

    assert [step.name for step in steps] == [
        "execution_data_audit",
        "fillability_gate",
        "signal_diagnostics",
        "fill_proxy_sweep",
        "fill_aware_selection",
        "candidate_gate",
        "overlay_profit_protect",
        "overlay_diagnostics",
        "summary",
    ]
    assert "pred.csv" in steps[0].command
    assert "board.duckdb" in steps[0].command
    assert "main_nav.csv" in steps[6].command


def test_run_steps_dry_run_does_not_execute_subprocess(capsys) -> None:
    steps = build_refresh_steps(
        predictions=Path("pred.csv"),
        board_db=Path("board.duckdb"),
        main_nav=Path("main_nav.csv"),
    )[:1]

    run_steps(steps, dry_run=True)

    out = capsys.readouterr().out
    assert "[board-refresh] execution_data_audit" in out
