from __future__ import annotations

import json

import duckdb
import pytest

from ml_stock_selector.constants import FEATURE_SET_BASELINE_A
from ml_stock_selector.feature_mart_batch import feature_mart_period_batches, run_feature_mart_batch
from tests.ml_fixtures import create_alpha_data_db, create_vpa_db, normalized_bars
from vpa_structure_recognizer.batch_runner import BatchPeriod


def test_feature_mart_period_batches_can_group_multiple_months() -> None:
    batches = feature_mart_period_batches("2018-01-01", "2018-05-15", warmup_months=13, batch_months=3)

    assert batches == [
        BatchPeriod("2016-12-01", "2018-01-01", "2018-03-31"),
        BatchPeriod("2017-03-01", "2018-04-01", "2018-05-15"),
    ]


def test_feature_mart_batch_uses_warmup_and_lookahead_but_writes_output_window(tmp_path):
    alpha_db = create_alpha_data_db(tmp_path / "alpha.duckdb", normalized_bars())
    vpa_db = create_vpa_db(tmp_path / "vpa.duckdb")
    ml_db = tmp_path / "ml.duckdb"

    counts = run_feature_mart_batch(
        alpha_data_db=str(alpha_db),
        vpa_db=str(vpa_db),
        ml_db=str(ml_db),
        normalized_bars_table="stock_bar_normalized_daily",
        batch=BatchPeriod("2024-01-02", "2024-01-03", "2024-01-05"),
        feature_set_id=FEATURE_SET_BASELINE_A,
        windows=[2],
        exclude_industry_metadata_from_features_json=True,
        lookahead_days=10,
    )

    assert counts["feature_mart"] == 12
    con = duckdb.connect(str(ml_db), read_only=True)
    try:
        feature_dates = con.execute(
            "select min(trade_date), max(trade_date), count(*) from ml_feature_mart_daily"
        ).fetchone()
        tradeability_dates = con.execute(
            "select min(trade_date), max(trade_date), count(*) from ml_tradeability_daily"
        ).fetchone()
        warmup_feature_json = con.execute(
            """
            select features_json
            from ml_feature_mart_daily
            where trade_date = '2024-01-03' and code = '000001.SZ'
            """
        ).fetchone()[0]
        next_trade_date, can_buy_next_open = con.execute(
            """
            select next_trade_date, can_buy_next_open
            from ml_tradeability_daily
            where trade_date = '2024-01-05' and code = '000001.SZ'
            """
        ).fetchone()
    finally:
        con.close()

    assert feature_dates == ("2024-01-03", "2024-01-05", 12)
    assert tradeability_dates == ("2024-01-03", "2024-01-05", 12)
    assert next_trade_date == "2024-01-08"
    assert can_buy_next_open is True
    features = json.loads(warmup_feature_json)
    assert features["ret_1d"] == pytest.approx(10.2 / 10.0 - 1.0)
