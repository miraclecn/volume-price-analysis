from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.config import load_ml_config
from ml_stock_selector.registry import activate_model_bundle
from ml_stock_selector.storage import init_ml_db


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--config", default="config/ml_default.toml")
    parser.add_argument("--ml-db")
    parser.add_argument("--confirm", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.confirm:
        raise SystemExit("--confirm is required to activate a model bundle")
    ml_db = args.ml_db
    if ml_db is None:
        config = load_ml_config(args.config)
        ml_db = str(config.data["ml_db"])
    con = init_ml_db(ml_db)
    try:
        activate_model_bundle(con, args.bundle_id)
    finally:
        con.close()
    print(f"activated_bundle_id={args.bundle_id}")


if __name__ == "__main__":
    main()
