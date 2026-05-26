from __future__ import annotations

import argparse

from vpa_structure_recognizer.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VPA structure recognizer")
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--as-of-date")
    parser.add_argument("--source")
    parser.add_argument("--output-db")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    result = run_pipeline(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        as_of_date=args.as_of_date,
        source=args.source,
        output_db=args.output_db,
        output_dir=args.output_dir,
    )
    print(f"output_db={result.output_db}")
    print(f"report_path={result.report_path}")
    print(f"table_counts={result.table_counts}")


if __name__ == "__main__":
    main()
