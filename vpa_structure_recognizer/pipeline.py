from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from vpa_structure_recognizer.backtest_validator import compute_validation_metrics
from vpa_structure_recognizer.bar_labeler import label_bars
from vpa_structure_recognizer.config import load_config
from vpa_structure_recognizer.data_sources import AuditedStockDuckDB, ResearchSourceDuckDB
from vpa_structure_recognizer.excel_exporter import export_excel_report
from vpa_structure_recognizer.feature_engineering import compute_features
from vpa_structure_recognizer.market_aggregates import build_market_bars, build_sector_bars
from vpa_structure_recognizer.sequence_analyzer import analyze_sequences
from vpa_structure_recognizer.state_classifier import classify_structure_states
from vpa_structure_recognizer.storage import init_vpa_db, upsert_dataframe
from vpa_structure_recognizer.top_down_ranker import rank_top_down
from vpa_structure_recognizer.trend_context import compute_trend_context


@dataclass(frozen=True)
class PipelineResult:
    output_db: Path
    report_path: Path
    table_counts: dict[str, int]


def run_pipeline(
    *,
    config_path: Path | str,
    start_date: str,
    end_date: str,
    source: str | None = None,
    output_db: Path | str | None = None,
    output_dir: Path | str | None = None,
    as_of_date: str | None = None,
) -> PipelineResult:
    config = load_config(config_path)
    output_db_path = Path(output_db or config.outputs["database"])
    report_dir = Path(output_dir or config.outputs["reports_dir"])
    source_reader = _source_reader(config.sources, source)

    stock_bars = source_reader.fetch_stock_bars(start_date, end_date)
    sector_bars = build_sector_bars(stock_bars)
    market_bars = build_market_bars(stock_bars)
    features = pd.concat(
        [
            compute_features(stock_bars, config.windows, "stock", scope_id_column="code"),
            compute_features(sector_bars, config.windows, "sector", scope_id_column="sector_code"),
            compute_features(
                _market_feature_input(market_bars),
                config.windows,
                "market",
                scope_id="ALL_A",
            ),
        ],
        ignore_index=True,
    )
    trend_context = compute_trend_context(features, config.parent_windows)
    labels = label_bars(features, config.parent_windows)
    sequences = analyze_sequences(labels, trend_context)
    states = classify_structure_states(sequences, trend_context)
    stock_sector_map = _stock_sector_map(stock_bars)
    ranked = rank_top_down(states, stock_sector_map, config.scoring_weights)

    con = init_vpa_db(output_db_path)
    upsert_dataframe(con, "vpa_features", features, ["date", "scope_type", "scope_id", "window_n"])
    upsert_dataframe(
        con,
        "vpa_trend_context",
        trend_context,
        ["date", "scope_type", "scope_id", "window_n"],
    )
    upsert_dataframe(
        con,
        "vpa_bar_context_labels",
        labels,
        ["date", "scope_type", "scope_id", "window_n"],
    )
    upsert_dataframe(
        con,
        "vpa_sequence_stats",
        sequences,
        ["date", "scope_type", "scope_id", "window_n"],
    )
    upsert_dataframe(con, "vpa_structure_state", ranked, ["date", "scope_type", "scope_id"])
    table_counts = {
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

    stock_states = ranked[ranked["scope_type"] == "stock"].copy()
    stock_states["sector_id"] = stock_states["scope_id"].map(stock_sector_map)
    validation = compute_validation_metrics(stock_states, stock_bars)
    report_date = (as_of_date or end_date).replace("-", "")
    report_path = report_dir / f"vpa_structure_report_{report_date}.xlsx"
    export_excel_report(
        report_path,
        ranked[ranked["scope_type"] == "market"],
        ranked[ranked["scope_type"] == "sector"],
        ranked[ranked["scope_type"] == "stock"],
        labels[labels["scope_type"] == "stock"],
        validation,
    )
    return PipelineResult(output_db=output_db_path, report_path=report_path, table_counts=table_counts)


def _source_reader(sources: dict[str, str], source: str | None) -> ResearchSourceDuckDB | AuditedStockDuckDB:
    selected = source or sources["research_source"]
    if selected in sources:
        selected = sources[selected]
    path = Path(selected)
    if path.name == "stock_data_audited.duckdb":
        return AuditedStockDuckDB(path)
    return ResearchSourceDuckDB(path)


def _market_feature_input(market_bars: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": market_bars["date"],
            "open": market_bars["all_a_equal_weight_open"],
            "high": market_bars["all_a_equal_weight_high"],
            "low": market_bars["all_a_equal_weight_low"],
            "close": market_bars["all_a_equal_weight_close"],
            "volume": market_bars["total_volume"],
            "amount": market_bars["total_amount"],
        }
    )
    frame["prev_close"] = frame["close"].shift(1)
    return frame


def _stock_sector_map(stock_bars: pd.DataFrame) -> dict[str, str]:
    latest = stock_bars.sort_values("date").dropna(subset=["industry_code"])
    return latest.groupby("code", sort=False)["industry_code"].last().to_dict()
