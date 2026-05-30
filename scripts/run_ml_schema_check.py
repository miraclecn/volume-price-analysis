from __future__ import annotations

import argparse
from pathlib import Path
import sys

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.contracts.vpa_schema import assert_vpa_schema_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vpa-db", required=True)
    args = parser.parse_args()
    con = duckdb.connect(args.vpa_db, read_only=True)
    try:
        assert_vpa_schema_contract(con)
    finally:
        con.close()
    print("vpa schema contract ok")


if __name__ == "__main__":
    main()
