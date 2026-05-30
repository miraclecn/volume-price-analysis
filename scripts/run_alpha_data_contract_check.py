from __future__ import annotations

import argparse
from pathlib import Path
import sys

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.contracts.alpha_data_contract import assert_alpha_data_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--table", default="stock_bar_normalized_daily")
    args = parser.parse_args()
    con = duckdb.connect(args.db, read_only=True)
    try:
        assert_alpha_data_contract(con, args.table)
    finally:
        con.close()
    print("alpha-data contract ok")


if __name__ == "__main__":
    main()
