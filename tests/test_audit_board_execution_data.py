from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from scripts.audit_board_execution_data import audit_board_execution_data


def test_audit_board_execution_data_reports_missing_database(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    _predictions().to_csv(predictions, index=False)

    audit = audit_board_execution_data(predictions_path=predictions, board_db=tmp_path / "missing.duckdb")

    assert audit["ok"] is False
    assert "missing board execution database" in audit["reason"]


def test_audit_board_execution_data_reports_top_signal_fill_coverage(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    db = tmp_path / "board.duckdb"
    _predictions().to_csv(predictions, index=False)
    _write_board_db(db)

    audit = audit_board_execution_data(predictions_path=predictions, board_db=db, top_n=2)

    assert audit["ok"] is True
    assert audit["events"]["matched_top_rows"] == 3
    assert audit["events"]["top_row_coverage"] == 0.75
    assert audit["events"]["missing_top_sample"] == [{"trade_date": "2024-01-03", "code": "e"}]
    assert audit["fills"]["matched_top_rows"] == 3
    assert audit["fills"]["avg_matched_fills_per_top_day"] == 1.5
    assert audit["fills"]["days_with_at_least_2_matched_fills"] == 1


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "code": "a", "pred_ret": 0.05, "pred_win_prob": 0.9},
            {"trade_date": "2024-01-02", "code": "b", "pred_ret": 0.04, "pred_win_prob": 0.8},
            {"trade_date": "2024-01-02", "code": "c", "pred_ret": 0.03, "pred_win_prob": 0.7},
            {"trade_date": "2024-01-03", "code": "d", "pred_ret": 0.05, "pred_win_prob": 0.9},
            {"trade_date": "2024-01-03", "code": "e", "pred_ret": 0.04, "pred_win_prob": 0.8},
            {"trade_date": "2024-01-03", "code": "f", "pred_ret": 0.03, "pred_win_prob": 0.7},
        ]
    )


def _write_board_db(path: Path) -> None:
    con = duckdb.connect(str(path))
    try:
        con.execute((Path(__file__).resolve().parents[1] / "sql" / "create_board_execution_tables.sql").read_text(encoding="utf-8"))
        con.executemany(
            """
            insert into board_intraday_events
            (trade_date, code, first_limit_time, last_limit_time, seal_duration_seconds, reopen_count, limit_up, close, is_close_sealed)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("2024-01-02", "a", "10:01", "14:55", 3600, 0, 11.0, 11.0, True),
                ("2024-01-02", "b", "10:02", "14:55", 3000, 1, 11.0, 11.0, True),
                ("2024-01-03", "d", "10:01", "14:55", 3600, 0, 11.0, 11.0, True),
            ],
        )
        con.executemany(
            """
            insert into board_order_fills
            (trade_date, code, signal_time, order_time, side, order_price, order_qty, filled_qty, avg_fill_price, status)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("2024-01-02", "a", "10:01", "10:01:01", "buy", 11.0, 1000, 500, 11.0, "partial"),
                ("2024-01-02", "b", "10:02", "10:02:01", "buy", 11.0, 1000, 500, 11.0, "partial"),
                ("2024-01-03", "d", "10:01", "10:01:01", "buy", 11.0, 1000, 500, 11.0, "partial"),
            ],
        )
    finally:
        con.close()
