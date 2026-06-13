from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb

from ml_stock_selector.feature_store import export_feature_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-db", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--row-group-size", type=int, default=50000)
    parser.add_argument("--compression", default="zstd")
    args = parser.parse_args()

    con = duckdb.connect(args.ml_db, read_only=True)
    try:
        result = export_feature_store(
            con,
            args.output_dir,
            args.dataset_version,
            args.feature_set_id,
            args.start_date,
            args.end_date,
            chunk_size=args.chunk_size,
            row_group_size=args.row_group_size,
            compression=args.compression,
            isolate_month_exports=True,
        )
    finally:
        con.close()
    print(f"rows={result.row_count} schema={result.feature_schema_path} metadata={result.metadata_path}")


if __name__ == "__main__":
    main()
