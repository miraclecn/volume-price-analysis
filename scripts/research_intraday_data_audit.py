from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_DUCKDB_PATHS = [
    "/home/nan/alpha-data-local/output/raw.duckdb",
    "/home/nan/alpha-data-local/output/research_source.duckdb",
    "/home/nan/alpha-data-local/output/pit_reference_staging.duckdb",
    "outputs/ml/ml_ret5_alpha_risk_20260619.duckdb",
    "outputs/vpa.duckdb",
]

INTRADAY_KEYWORDS = [
    "minute",
    "min1",
    "1min",
    "tick",
    "order_book",
    "level2",
    "level_2",
    "l2",
    "bid",
    "ask",
    "auction",
    "seal",
    "queue",
    "intraday",
]

DAILY_LIMIT_KEYWORDS = [
    "limit",
    "up_limit",
    "down_limit",
    "is_limit_up_open_lock",
    "is_limit_down_open_lock",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", nargs="*", default=DEFAULT_DUCKDB_PATHS)
    parser.add_argument("--out-json", default="outputs/limit_hit_research/reports/intraday_data_audit_20260622.json")
    args = parser.parse_args()
    report = audit_duckdb_paths([Path(path) for path in args.duckdb])
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {out}")


def audit_duckdb_paths(paths: list[Path]) -> dict[str, object]:
    databases = []
    intraday_hits: list[dict[str, object]] = []
    daily_limit_hits: list[dict[str, object]] = []
    for path in paths:
        db_report = {"path": str(path), "exists": path.exists(), "tables": []}
        if not path.exists():
            databases.append(db_report)
            continue
        con = duckdb.connect(str(path), read_only=True)
        try:
            tables = con.execute(
                "select table_schema, table_name from information_schema.tables order by table_schema, table_name"
            ).fetchdf()
            for row in tables.itertuples(index=False):
                table_ref = f"{row.table_schema}.{row.table_name}"
                cols = con.execute(f'describe "{row.table_schema}"."{row.table_name}"').fetchdf()["column_name"].astype(str).tolist()
                table_hit = _keyword_hits(table_ref, INTRADAY_KEYWORDS)
                intraday_col_hits = [col for col in cols if _keyword_hits(col, INTRADAY_KEYWORDS)]
                daily_limit_col_hits = [col for col in cols if _keyword_hits(col, DAILY_LIMIT_KEYWORDS)]
                table_summary = {
                    "table": table_ref,
                    "column_count": len(cols),
                    "intraday_table_keywords": table_hit,
                    "intraday_column_hits": intraday_col_hits,
                    "daily_limit_column_hits": daily_limit_col_hits,
                }
                if table_hit or intraday_col_hits or daily_limit_col_hits:
                    db_report["tables"].append(table_summary)
                if table_hit or intraday_col_hits:
                    intraday_hits.append({"database": str(path), **table_summary})
                if daily_limit_col_hits:
                    daily_limit_hits.append({"database": str(path), **table_summary})
        finally:
            con.close()
        databases.append(db_report)
    required_intraday_fields = [
        "first_limit_time",
        "seal_duration",
        "reopen_count",
        "limit_order_queue",
        "bid_ask_depth",
        "auction_imbalance",
        "actual_fillable_at_limit",
    ]
    return {
        "summary": {
            "database_count": len(databases),
            "intraday_hit_count": len(intraday_hits),
            "daily_limit_hit_count": len(daily_limit_hits),
            "has_intraday_execution_data": bool(_credible_intraday_hits(intraday_hits)),
            "required_missing_fields": required_intraday_fields,
            "daily_available_but_not_enough": bool(daily_limit_hits),
        },
        "databases": databases,
        "intraday_hits": intraday_hits,
        "daily_limit_hits": daily_limit_hits,
    }


def _credible_intraday_hits(hits: list[dict[str, object]]) -> list[dict[str, object]]:
    credible = []
    credible_table_terms = ["minute", "min1", "1min", "tick", "level2", "level_2", "order_book", "auction", "intraday"]
    credible_column_terms = ["bid", "ask", "first_limit", "seal", "reopen", "queue_volume", "queue_amount", "order_imbalance"]
    for hit in hits:
        table = str(hit.get("table", ""))
        columns = [str(col) for col in hit.get("intraday_column_hits", [])]
        table_lower = table.lower()
        if any(term in table_lower for term in credible_table_terms):
            credible.append(hit)
            continue
        if any(any(term in col.lower() for term in credible_column_terms) for col in columns):
            credible.append(hit)
    return credible


def _keyword_hits(text: str, keywords: list[str]) -> list[str]:
    lower = text.lower()
    return [keyword for keyword in keywords if keyword in lower]


if __name__ == "__main__":
    main()
