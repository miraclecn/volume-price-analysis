from pathlib import Path

import duckdb

from vpa_structure_recognizer.pipeline import run_pipeline


def create_unknown_industry_source(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        """
        create table stock_bar_normalized_daily (
            trade_date varchar,
            code varchar,
            open double,
            high double,
            low double,
            close double,
            prev_close double,
            volume double,
            amount double,
            turnover_rate double,
            is_st boolean,
            is_paused boolean,
            limit_up double,
            limit_down double,
            industry_code varchar,
            industry_name varchar
        )
        """
    )

    daily_rows = []
    closes_by_code = {
        "000001.SZ": [10.0, 10.3, 10.6, 10.4, 10.9, 11.2, 11.0, 11.5],
        "000002.SZ": [20.0, 19.8, 20.2, 20.5, 20.4, 20.9, 21.2, 21.0],
    }
    dates = [
        "20240102",
        "20240103",
        "20240104",
        "20240105",
        "20240108",
        "20240109",
        "20240110",
        "20240111",
    ]
    for code, closes in closes_by_code.items():
        for idx, (date, close) in enumerate(zip(dates, closes)):
            industry_code = "BK001" if code == "000001.SZ" else "UNKNOWN"
            industry_name = "Banking" if code == "000001.SZ" else "UNKNOWN"
            daily_rows.append(
                (
                    date,
                    code,
                    close - 0.2,
                    close + 0.4,
                    close - 0.5,
                    close,
                    close - 0.1,
                    1000.0 + idx * 100,
                    close * (1000.0 + idx * 100),
                    1.0,
                    False,
                    False,
                    close * 1.1,
                    close * 0.9,
                    industry_code,
                    industry_name,
                )
            )

    con.executemany(
        "insert into stock_bar_normalized_daily values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        daily_rows,
    )
    con.close()


def test_pipeline_accepts_unknown_industry_contract_and_keeps_stock_outputs(tmp_path):
    source_db = tmp_path / "research.duckdb"
    output_db = tmp_path / "vpa.duckdb"
    create_unknown_industry_source(source_db)

    run_pipeline(
        config_path="config/default.toml",
        start_date="2024-01-02",
        end_date="2024-01-11",
        source=str(source_db),
        output_db=output_db,
        output_dir=tmp_path / "reports",
    )

    con = duckdb.connect(str(output_db))
    try:
        stock_counts = {
            table: con.execute(
                f"""
                select count(*)
                from {table}
                where scope_type = 'stock' and scope_id = '000002.SZ'
                """
            ).fetchone()[0]
            for table in [
                "vpa_features",
                "vpa_bar_context_labels",
                "vpa_sequence_stats",
                "vpa_structure_state",
            ]
        }
    finally:
        con.close()

    assert all(count > 0 for count in stock_counts.values())
