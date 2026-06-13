from __future__ import annotations

import json
from pathlib import Path
import os

import pytest
from ml_stock_selector.serving.live_sim import init_live_sim_db
from ml_stock_selector.serving.qmt_order_export import export_qmt_orders


def test_export_qmt_orders_writes_fixed_filename_with_buy_and_sell_dates(tmp_path: Path) -> None:
    con = init_live_sim_db(tmp_path / "live.duckdb")
    con.execute(
        """
        insert into live_sim_planned_orders
        (account_id, decision_date, execution_date, code, side, target_weight, trade_score_v2,
         absolute_rank_pct, active_rank_pct, risk_rank_pct, adv20_amount, estimated_price,
         estimated_qty, target_value, entry_reason, signal_action, status, generated_at)
        values
        ('paper', '2024-01-02', '2024-01-03', '000001.SZ', 'buy', 0.5, 0.9,
         0.9, 0.8, 0.2, 20000000, 10.25, 1200, 12300, 'core_pool', 'buy', 'planned', 'now'),
        ('paper', '2024-01-02', '2024-01-03', '600000.SH', 'sell', 0.0, null,
         null, null, null, null, 8.91, 1000, 0, null, 'sell', 'planned', 'now'),
        ('paper', '2024-01-02', '2024-01-03', '000002.SZ', 'hold', 0.5, 0.8,
         0.8, 0.7, 0.3, 18000000, 9.5, 0, 150000, null, 'hold', 'planned', 'now')
        """
    )

    output_path = export_qmt_orders(
        con,
        account_id="paper",
        decision_date="2024-01-02",
        output_json=tmp_path / "orders.json",
        generated_at="2024-01-02T18:00:00+08:00",
    )

    assert output_path == tmp_path / "orders.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "qmt_ml_order.v1"
    assert payload["source"] == "live_sim_daily"
    assert payload["account_id"] == "paper"
    assert payload["decision_date"] == "2024-01-02"
    assert payload["execution_date"] == "2024-01-03"
    assert payload["generated_at"] == "2024-01-02T18:00:00+08:00"
    assert [order["side"] for order in payload["orders"]] == ["buy", "sell"]
    assert payload["orders"][0]["client_order_id"] == "paper_20240102_000001_SZ_buy"
    assert payload["orders"][0]["buy_date"] == "2024-01-03"
    assert payload["orders"][0]["sell_date"] is None
    assert payload["orders"][0]["qty"] == 1200
    assert payload["orders"][0]["reference_price"] == 10.25
    assert payload["orders"][1]["buy_date"] is None
    assert payload["orders"][1]["sell_date"] == "2024-01-03"
    con.close()


def test_export_qmt_orders_optionally_copies_fixed_filename(tmp_path: Path) -> None:
    con = init_live_sim_db(tmp_path / "live.duckdb")
    con.execute(
        """
        insert into live_sim_planned_orders
        (account_id, decision_date, execution_date, code, side, target_weight, trade_score_v2,
         absolute_rank_pct, active_rank_pct, risk_rank_pct, adv20_amount, estimated_price,
         estimated_qty, target_value, entry_reason, signal_action, status, generated_at)
        values ('paper', '2024-01-02', '2024-01-03', '000001.SZ', 'buy', 0.5, 0.9,
                0.9, 0.8, 0.2, 20000000, 10.25, 1200, 12300, 'core_pool', 'buy', 'planned', 'now')
        """
    )

    export_qmt_orders(
        con,
        account_id="paper",
        decision_date="2024-01-02",
        output_json=tmp_path / "local" / "orders.json",
        copy_dir=tmp_path / "copy",
        generated_at="2024-01-02T18:00:00+08:00",
    )

    copied = tmp_path / "copy" / "orders.json"
    assert copied.exists()
    assert json.loads(copied.read_text(encoding="utf-8"))["orders"][0]["code"] == "000001.SZ"
    con.close()


def test_export_qmt_orders_keeps_execution_date_when_no_orders(tmp_path: Path) -> None:
    con = init_live_sim_db(tmp_path / "live.duckdb")

    output_path = export_qmt_orders(
        con,
        account_id="paper",
        decision_date="2024-01-02",
        execution_date="2024-01-03",
        output_json=tmp_path / "orders.json",
        generated_at="2024-01-02T18:00:00+08:00",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["execution_date"] == "2024-01-03"
    assert payload["orders"] == []
    con.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only UNC guard")
def test_export_qmt_orders_rejects_raw_unc_copy_dir_on_posix(tmp_path: Path) -> None:
    con = init_live_sim_db(tmp_path / "live.duckdb")

    with pytest.raises(ValueError, match="UNC"):
        export_qmt_orders(
            con,
            account_id="paper",
            decision_date="2024-01-02",
            execution_date="2024-01-03",
            output_json=tmp_path / "orders.json",
            copy_dir=r"\\DESKTOP-2DOUG97\Users\chen\qmt_ml_order\orders",
            generated_at="2024-01-02T18:00:00+08:00",
        )
    con.close()
