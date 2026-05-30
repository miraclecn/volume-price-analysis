from pathlib import Path

import duckdb

from vpa_structure_recognizer.pipeline import run_pipeline


def _create_source(path: Path) -> None:
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
    rows = []
    for idx, (date, close) in enumerate(
        [
            ("20240102", 10.0),
            ("20240103", 10.5),
            ("20240104", 10.2),
            ("20240105", 11.0),
            ("20240108", 11.4),
            ("20240109", 11.2),
        ]
    ):
        rows.append(
            (
                date,
                "000001.SZ",
                close - 0.2,
                close + 0.4,
                close - 0.5,
                close,
                close - 0.1,
                1000 + idx * 100,
                close * (1000 + idx * 100),
                1.0,
                False,
                False,
                close * 1.1,
                close * 0.9,
                "BK001",
                "Banking",
            )
        )
    con.executemany(
        "insert into stock_bar_normalized_daily values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.close()


def test_pipeline_smoke_populates_vpa_tables_and_report(tmp_path):
    source_db = tmp_path / "research.duckdb"
    output_db = tmp_path / "vpa.duckdb"
    output_dir = tmp_path / "reports"
    _create_source(source_db)

    result = run_pipeline(
        config_path="config/default.toml",
        start_date="2024-01-02",
        end_date="2024-01-09",
        source=str(source_db),
        output_db=output_db,
        output_dir=output_dir,
    )

    con = duckdb.connect(str(output_db))
    counts = {
        table: con.execute(f"select count(*) from {table}").fetchone()[0]
        for table in [
            "vpa_features",
            "vpa_trend_context",
            "vpa_bar_context_labels",
            "vpa_sequence_stats",
            "vpa_structure_state",
        ]
    }
    con.close()

    assert all(count > 0 for count in counts.values())
    assert result.report_path.exists()
    assert result.output_db == output_db
