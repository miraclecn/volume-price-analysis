from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from ml_stock_selector.fundamental_features import (
    FUNDAMENTAL_FEATURE_COLUMNS,
    append_fundamental_columns,
    load_fundamental_features_for_metadata,
)


def test_load_fundamental_features_preserves_metadata_order_and_fills_missing(tmp_path):
    raw_db = tmp_path / "raw.duckdb"
    con = duckdb.connect(raw_db)
    try:
        con.execute(
            """
            create table mart_fundamental (
                code varchar,
                date varchar,
                eps double,
                roe double,
                roa double,
                gross_margin double,
                netprofit_margin double,
                current_ratio double,
                debt_to_assets double,
                revenue_ps double,
                netprofit_yoy double,
                dt_netprofit_yoy double,
                or_yoy double,
                q_sales_yoy double,
                assets_yoy double,
                equity_yoy double
            )
            """
        )
        con.execute(
            """
            insert into mart_fundamental values
            ('000002.SZ', '20240103', 0.20, 0.12, 0.05, 0.31, 0.08, 1.60, 0.42, 2.10, 0.15, 0.13, 0.11, 0.09, 0.07, 0.06),
            ('000001.SZ', '20240102', null, 0.10, 0.04, 0.30, 0.07, 1.50, 0.40, 2.00, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05)
            """
        )
    finally:
        con.close()

    metadata = pd.DataFrame(
        {
            "trade_date": ["2024-01-03", "2024-01-02", "2024-01-04"],
            "code": ["000002.SZ", "000001.SZ", "000003.SZ"],
        }
    )

    features = load_fundamental_features_for_metadata(raw_db, metadata)

    assert list(features.columns) == [target for _, target in FUNDAMENTAL_FEATURE_COLUMNS]
    assert features.shape == (3, len(FUNDAMENTAL_FEATURE_COLUMNS))
    assert features.loc[0, "fund_eps"] == np.float32(0.20)
    assert features.loc[0, "fund_or_yoy"] == np.float32(0.11)
    assert features.loc[1, "fund_eps"] == np.float32(0.0)
    assert features.loc[2].eq(0.0).all()
    assert features.dtypes.eq(np.float32).all()


def test_append_fundamental_columns_returns_augmented_matrix_and_column_names(tmp_path):
    raw_db = tmp_path / "raw.duckdb"
    con = duckdb.connect(raw_db)
    try:
        con.execute(
            """
            create table mart_fundamental as
            select
                '000001.SZ'::varchar as code,
                '20240102'::varchar as date,
                0.20::double as eps,
                0.12::double as roe,
                0.05::double as roa,
                0.31::double as gross_margin,
                0.08::double as netprofit_margin,
                1.60::double as current_ratio,
                0.42::double as debt_to_assets,
                2.10::double as revenue_ps,
                0.15::double as netprofit_yoy,
                0.13::double as dt_netprofit_yoy,
                0.11::double as or_yoy,
                0.09::double as q_sales_yoy,
                0.07::double as assets_yoy,
                0.06::double as equity_yoy
            """
        )
    finally:
        con.close()

    base = np.array([[1.0, 2.0]], dtype=np.float32)
    metadata = pd.DataFrame({"trade_date": ["2024-01-02"], "code": ["000001.SZ"]})

    augmented, columns = append_fundamental_columns(base, metadata, raw_db)

    assert augmented.shape == (1, base.shape[1] + len(FUNDAMENTAL_FEATURE_COLUMNS))
    assert augmented.dtype == np.float32
    assert np.allclose(augmented[0, :2], base[0])
    assert columns[0] == "fund_eps"
    assert columns[-1] == "fund_equity_yoy"
    assert augmented[0, 2] == np.float32(0.20)
