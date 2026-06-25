from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from scripts.research_board_fillability_gate import (
    FillabilityGateConfig,
    evaluate_fillability,
    run_fillability_gate,
)


def test_fillability_gate_passes_when_top_two_fill_with_good_quality() -> None:
    predictions = _predictions()
    fills = pd.DataFrame(
        [
            _fill("2024-01-02", "a1"),
            _fill("2024-01-02", "a2"),
            _fill("2024-01-03", "b1"),
            _fill("2024-01-03", "b2"),
        ]
    )

    result = evaluate_fillability(predictions, fills, FillabilityGateConfig())
    summary = result["summary"]

    assert summary["ok"] is True
    assert summary["avg_fills_per_active_day"] == pytest.approx(2.0)
    assert summary["fill_count_ok"] is True
    assert summary["quality_ok"] is True


def test_fillability_gate_fails_when_fills_are_worse_than_adverse_benchmark() -> None:
    predictions = _predictions()
    fills = pd.DataFrame(
        [
            _fill("2024-01-02", "a4", avg_fill_price=10.2),
            _fill("2024-01-02", "a5", avg_fill_price=10.2),
            _fill("2024-01-03", "b4", avg_fill_price=10.2),
            _fill("2024-01-03", "b5", avg_fill_price=10.2),
        ]
    )

    result = evaluate_fillability(predictions, fills, FillabilityGateConfig())
    summary = result["summary"]

    assert summary["ok"] is False
    assert summary["fill_count_ok"] is True
    assert summary["quality_ok"] is False
    assert "quality worse" in summary["reason"]


def test_run_fillability_gate_reports_missing_fills_table(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    _predictions().to_csv(predictions_path, index=False)
    board_db = tmp_path / "board_execution.duckdb"
    con = duckdb.connect(str(board_db))
    try:
        con.execute("create table board_intraday_events (trade_date varchar, code varchar)")
    finally:
        con.close()

    result = run_fillability_gate(
        predictions_path=predictions_path,
        board_db=board_db,
        out_dir=tmp_path / "reports",
    )

    assert result["summary"]["ok"] is False
    assert result["summary"]["reason"] == "missing board_order_fills"
    assert (tmp_path / "reports" / "board_fillability_gate_summary.json").exists()


def _predictions() -> pd.DataFrame:
    rows = []
    for date, prefix in [("2024-01-02", "a"), ("2024-01-03", "b")]:
        realized = [0.08, 0.06, 0.03, -0.02, -0.03]
        for idx, ret in enumerate(realized, start=1):
            rows.append(
                {
                    "trade_date": date,
                    "code": f"{prefix}{idx}",
                    "target_ret_net": ret,
                    "pred_ret": 0.10 - idx * 0.01,
                    "pred_win_prob": 0.90 - idx * 0.05,
                }
            )
    return pd.DataFrame(rows)


def _fill(trade_date: str, code: str, *, avg_fill_price: float = 10.0) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "code": code,
        "signal_time": "14:55:00",
        "order_time": "14:55:01",
        "side": "buy",
        "order_price": 10.0,
        "order_qty": 1000,
        "filled_qty": 1000,
        "avg_fill_price": avg_fill_price,
        "status": "filled",
    }
