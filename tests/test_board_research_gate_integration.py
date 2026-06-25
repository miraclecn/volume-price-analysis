from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from scripts.research_board_candidate_gate import build_candidate_manifest
from scripts.research_board_fillability_gate import run_fillability_gate


def test_board_research_gate_can_promote_when_real_fills_pass(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    board_db = tmp_path / "board_execution.duckdb"
    out_dir = tmp_path / "fillability"
    metrics = tmp_path / "selection_metrics.csv"
    profile = tmp_path / "profile.csv"
    _predictions().to_csv(predictions, index=False)
    _write_board_execution_db(board_db)
    _selection_metrics().to_csv(metrics, index=False)
    _profile().to_csv(profile, index=False)

    fillability = run_fillability_gate(
        predictions_path=predictions,
        board_db=board_db,
        out_dir=out_dir,
    )
    manifest = build_candidate_manifest(
        fillability_summary_path=out_dir / "board_fillability_gate_summary.json",
        selection_metrics_path=metrics,
        signal_profile_path=profile,
        candidate_variant="neutral_expected_pred_ret",
        min_live_fills_per_day=2.0,
    )

    assert fillability["summary"]["ok"] is True
    assert fillability["summary"]["avg_fills_per_active_day"] == 2.0
    assert manifest["status"] == "live_candidate"
    assert manifest["promotion_gate"]["blockers"] == []


def _predictions() -> pd.DataFrame:
    rows = []
    for date, prefix in [("2024-01-02", "a"), ("2024-01-03", "b")]:
        for idx, ret in enumerate([0.08, 0.06, -0.02], start=1):
            rows.append(
                {
                    "trade_date": date,
                    "code": f"{prefix}{idx}",
                    "target_ret_net": ret,
                    "target_win": int(ret > 0),
                    "second_board_success": ret > 0.05,
                    "pred_ret": 0.10 - idx * 0.01,
                    "pred_win_prob": 0.90 - idx * 0.05,
                }
            )
    return pd.DataFrame(rows)


def _write_board_execution_db(path: Path) -> None:
    con = duckdb.connect(str(path))
    try:
        con.execute((Path(__file__).resolve().parents[1] / "sql" / "create_board_execution_tables.sql").read_text(encoding="utf-8"))
        fills = []
        for date, prefix in [("2024-01-02", "a"), ("2024-01-03", "b")]:
            for idx in [1, 2]:
                fills.append((date, f"{prefix}{idx}", "14:55:00", "14:55:01", "buy", 10.0, 1000, 1000, 10.0, "filled"))
        con.executemany(
            """
            insert into board_order_fills
            (trade_date, code, signal_time, order_time, side, order_price, order_qty, filled_qty, avg_fill_price, status)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            fills,
        )
    finally:
        con.close()


def _selection_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant": "neutral_expected_pred_ret",
                "annual_return": 0.35,
                "total_return": 4.3,
                "max_drawdown": -0.01,
                "sharpe": 2.0,
                "avg_expected_fills_per_day": 2.9,
                "avg_expected_exposure": 0.029,
                "selected_return_mean": 0.05,
                "selected_turnover_mean": 9.0,
                "max_candidates": 5,
                "name_weight": 0.01,
                "total_exposure_cap": 0.05,
            }
        ]
    )


def _profile() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"universe": "all_sealed", "rows": 6, "target_ret_net_mean": 0.04, "win_rate": 0.67, "second_board_rate": 0.33},
            {"universe": "top_candidate", "rows": 4, "target_ret_net_mean": 0.07, "win_rate": 1.0, "second_board_rate": 1.0},
        ]
    )
