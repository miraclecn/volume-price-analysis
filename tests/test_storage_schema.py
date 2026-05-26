import duckdb
import pandas as pd

from vpa_structure_recognizer.storage import init_vpa_db, upsert_dataframe


def test_init_vpa_db_creates_project_owned_tables(tmp_path):
    db_path = tmp_path / "vpa.duckdb"

    init_vpa_db(db_path)

    con = duckdb.connect(str(db_path))
    tables = {
        row[0]
        for row in con.execute(
            """
            select table_name
            from information_schema.tables
            where table_schema = 'main'
            """
        ).fetchall()
    }
    con.close()

    assert {
        "vpa_features",
        "vpa_trend_context",
        "vpa_bar_context_labels",
        "vpa_sequence_stats",
        "vpa_structure_state",
    }.issubset(tables)


def test_upsert_dataframe_replaces_existing_feature_key(tmp_path):
    db_path = tmp_path / "vpa.duckdb"
    con = init_vpa_db(db_path)

    first = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "scope_type": "stock",
                "scope_id": "000001.SZ",
                "window_n": 20,
                "open": 10.0,
                "high": 11.0,
                "low": 9.8,
                "close": 10.5,
                "prev_close": 10.0,
                "volume": 1000.0,
                "amount": 10500.0,
                "ret_pct": 0.05,
                "range_pct": 0.12,
                "body_pct": 0.05,
                "upper_shadow_pct": 0.05,
                "lower_shadow_pct": 0.02,
                "body_ratio": 0.4167,
                "upper_shadow_ratio": 0.4167,
                "lower_shadow_ratio": 0.1666,
                "close_position": 0.5833,
                "vol_ma_n": 1000.0,
                "vol_rvol_n": 1.0,
                "range_pct_ma_n": 0.12,
                "range_rvol_n": 1.0,
                "body_pct_ma_n": 0.05,
                "body_rvol_n": 1.0,
                "price_high_n": 11.0,
                "price_low_n": 9.8,
                "price_position_n": 0.5833,
                "ma_n": 10.5,
                "ma_slope_n": 0.0,
            }
        ]
    )
    second = first.copy()
    second.loc[0, "close"] = 10.8

    upsert_dataframe(con, "vpa_features", first, ["date", "scope_type", "scope_id", "window_n"])
    upsert_dataframe(con, "vpa_features", second, ["date", "scope_type", "scope_id", "window_n"])

    rows = con.execute("select count(*), max(close) from vpa_features").fetchone()
    con.close()

    assert rows == (1, 10.8)
