from pathlib import Path

import duckdb

from vpa_structure_recognizer.pipeline import run_pipeline


def _create_source(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        """
        create table daily_bar_pit (
            security_id varchar,
            trade_date varchar,
            is_st boolean,
            open_adj double,
            high_adj double,
            low_adj double,
            close_adj double,
            pre_close double,
            adj_factor double,
            volume_shares double,
            turnover_value_cny double,
            turnover_rate_pct double
        )
        """
    )
    con.execute(
        """
        create table tradeability_state_daily (
            security_id varchar,
            trade_date varchar,
            is_suspended boolean,
            up_limit double,
            down_limit double
        )
        """
    )
    con.execute(
        """
        create table industry_classification_pit (
            security_id varchar,
            industry_code varchar,
            industry_name varchar,
            effective_at varchar,
            removed_at varchar
        )
        """
    )
    rows = []
    tradeability = []
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
                "000001.SZ",
                date,
                False,
                close - 0.2,
                close + 0.4,
                close - 0.5,
                close,
                close - 0.1,
                1.0,
                1000 + idx * 100,
                close * (1000 + idx * 100),
                1.0,
            )
        )
        tradeability.append(("000001.SZ", date, False, close * 1.1, close * 0.9))
    con.executemany("insert into daily_bar_pit values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.executemany("insert into tradeability_state_daily values (?, ?, ?, ?, ?)", tradeability)
    con.execute(
        "insert into industry_classification_pit values ('000001.SZ', 'BK001', 'Banking', '20200101', null)"
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
