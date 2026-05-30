import duckdb
import pandas as pd

from tests.test_vpa_input_contract import create_unknown_industry_source
from vpa_structure_recognizer.market_aggregates import build_sector_bars
from vpa_structure_recognizer.pipeline import run_pipeline
from vpa_structure_recognizer.top_down_ranker import rank_top_down


def _state(scope_type, scope_id, final_state):
    return {
        "date": "2024-01-31",
        "scope_type": scope_type,
        "scope_id": scope_id,
        "final_state": final_state,
        "confidence": 0.8,
        "risk_flags": "",
        "main_features": final_state,
    }


def test_unknown_industry_stock_has_complete_stock_scope_pipeline_outputs(tmp_path):
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
        state = con.execute(
            """
            select sector_score, risk_flags
            from vpa_structure_state
            where scope_type = 'stock' and scope_id = '000002.SZ'
            order by date desc
            limit 1
            """
        ).fetchone()
        counts = {
            table: con.execute(
                f"""
                select count(distinct date)
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

    assert all(count > 0 for count in counts.values())
    assert state[0] == 50.0
    assert "industry_unknown" in state[1]


def test_unknown_industry_sector_aggregate_is_observational_group():
    stock_bars = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "code": "000002.SZ",
                "open": 20.0,
                "high": 20.8,
                "low": 19.8,
                "close": 20.5,
                "prev_close": 20.0,
                "volume": 2000.0,
                "amount": 41000.0,
                "is_st": False,
                "is_paused": False,
                "limit_up": 22.0,
                "limit_down": 18.0,
                "industry_code": "UNKNOWN",
                "industry_name": "UNKNOWN",
            }
        ]
    )

    sectors = build_sector_bars(stock_bars)

    assert sectors["sector_code"].tolist() == ["UNKNOWN"]
    assert sectors["sector_name"].tolist() == ["UNKNOWN"]
    assert sectors["member_count"].tolist() == [1]


def test_unknown_industry_does_not_use_unknown_sector_strength_for_stock_rating():
    states = pd.DataFrame(
        [
            _state("market", "ALL_A", "HEALTHY_UPTREND"),
            _state("sector", "UNKNOWN", "BREAKDOWN"),
            _state("stock", "000002.SZ", "HEALTHY_UPTREND"),
        ]
    )

    ranked = rank_top_down(states, {"000002.SZ": "UNKNOWN"})
    stock = ranked[ranked["scope_type"] == "stock"].iloc[0]

    assert stock["sector_score"] == 50.0
    assert stock["relative_strength_score"] == 45.0
    assert stock["final_rating"] == "B"
    assert "industry_unknown" in stock["risk_flags"]
    assert "逆势个股" not in stock["risk_flags"]
