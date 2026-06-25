from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_stock_selector.feature_store import compute_feature_schema_hash


FEATURE_SET_ID = "vpa_d_sequence"
WINDOWS = [5, 10, 20, 60, 120, 240]
LONG_WINDOWS = [20, 60, 120, 240]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-store-dir", default="outputs/ml/feature_store")
    parser.add_argument("--source-version", default="v2_fix12_prevext_20260616")
    parser.add_argument("--source-db", default="/home/nan/alpha-data-local/output/research_source.duckdb")
    parser.add_argument("--source-table", default="stock_bar_normalized_daily")
    parser.add_argument("--variant", action="append", choices=["mtd", "drop_long", "month_reset"])
    parser.add_argument("--version-suffix", default="20260618")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    variants = args.variant or ["mtd", "drop_long", "month_reset"]
    root = Path(args.feature_store_dir)
    source_root = root / f"dataset_version={args.source_version}" / f"feature_set_id={FEATURE_SET_ID}"
    source_schema = json.loads((source_root / "feature_schema.json").read_text(encoding="utf-8"))
    source_metadata = json.loads((source_root / "_metadata.json").read_text(encoding="utf-8"))
    source_numeric_columns = list(source_schema["numeric_columns"])

    for variant in variants:
        dataset_version = f"v2_fix12_{variant}_{args.version_suffix}"
        target_root = root / f"dataset_version={dataset_version}" / f"feature_set_id={FEATURE_SET_ID}"
        if target_root.exists():
            if not args.force:
                raise FileExistsError(f"{target_root} already exists; pass --force to overwrite")
            shutil.rmtree(target_root)
        target_root.mkdir(parents=True, exist_ok=True)
        numeric_columns = _numeric_columns_for_variant(variant, source_numeric_columns)
        _copy_variant_partitions(
            source_root,
            target_root,
            variant,
            dataset_version,
            numeric_columns,
            args.source_db,
            args.source_table,
        )
        schema_payload = _schema_payload(source_schema, dataset_version, numeric_columns)
        metadata_payload = _metadata_payload(source_metadata, dataset_version, variant, args.source_db)
        metadata_payload["schema_hash"] = schema_payload["schema_hash"]
        (target_root / "feature_schema.json").write_text(
            json.dumps(schema_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (target_root / "_metadata.json").write_text(
            json.dumps(metadata_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "variant": variant,
                    "dataset_version": dataset_version,
                    "root": str(target_root),
                    "numeric_columns": len(numeric_columns),
                    "row_count": metadata_payload.get("row_count"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


def _numeric_columns_for_variant(variant: str, source_numeric_columns: list[str]) -> list[str]:
    if variant == "drop_long":
        drop = set(_drop_long_columns())
        return [col for col in source_numeric_columns if col not in drop]
    if variant == "mtd":
        return source_numeric_columns + _mtd_columns()
    if variant == "month_reset":
        return source_numeric_columns + _month_reset_columns()
    raise ValueError(f"unknown variant: {variant}")


def _drop_long_columns() -> list[str]:
    families = [
        "ret_{w}d",
        "turnover_mean_{w}d",
        "amount_ratio_{w}d",
        "volume_ratio_{w}d",
        "high_distance_{w}d",
        "low_distance_{w}d",
        "volatility_{w}d",
    ]
    return [pattern.format(w=window) for window in LONG_WINDOWS for pattern in families]


def _mtd_columns() -> list[str]:
    return [
        "mtd_trading_day_in_month",
        "mtd_ret",
        "mtd_turnover_mean",
        "mtd_amount_ratio",
        "mtd_volume_ratio",
        "mtd_high_distance",
        "mtd_low_distance",
    ]


def _month_reset_columns() -> list[str]:
    columns = ["month_reset_ret_1d", "month_reset_open_gap_pct"]
    for window in WINDOWS:
        columns.extend(
            [
                f"month_reset_ret_{window}d",
                f"month_reset_turnover_mean_{window}d",
                f"month_reset_amount_ratio_{window}d",
                f"month_reset_volume_ratio_{window}d",
                f"month_reset_high_distance_{window}d",
                f"month_reset_low_distance_{window}d",
                f"month_reset_volatility_{window}d",
            ]
        )
    return columns


def _copy_variant_partitions(
    source_root: Path,
    target_root: Path,
    variant: str,
    dataset_version: str,
    numeric_columns: list[str],
    source_db: str,
    source_table: str,
) -> None:
    parquet_files = sorted(source_root.glob("year=*/month=*/part-00000.parquet"))
    con = duckdb.connect(":memory:")
    try:
        con.execute("set preserve_insertion_order = false")
        con.execute("set threads = 4")
        con.execute(f"attach '{_escape_sql_literal(source_db)}' as src (read_only)")
        for parquet_path in parquet_files:
            year = parquet_path.parent.parent.name.split("=", 1)[1]
            month = parquet_path.parent.name.split("=", 1)[1]
            out_dir = target_root / f"year={year}" / f"month={month}"
            out_dir.mkdir(parents=True, exist_ok=True)
            _copy_one_month(
                con,
                parquet_path,
                out_dir / "part-00000.parquet",
                variant,
                dataset_version,
                numeric_columns,
                source_table,
                year,
                month,
            )
            print(f"{variant} {year}-{month} done", flush=True)
    finally:
        con.close()


def _copy_one_month(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    output_path: Path,
    variant: str,
    dataset_version: str,
    numeric_columns: list[str],
    source_table: str,
    year: str,
    month: str,
) -> None:
    if variant == "drop_long":
        select_sql = _base_select_sql(dataset_version, numeric_columns)
        con.execute(
            f"""
            copy (
                {select_sql}
                from read_parquet('{_escape_sql_literal(str(parquet_path))}') as base
            ) to '{_escape_sql_literal(str(output_path))}' (format parquet, compression zstd, row_group_size 50000)
            """
        )
        return

    extra_select = _mtd_select_sql() if variant == "mtd" else _month_reset_select_sql()
    select_sql = _base_select_sql(dataset_version, [col for col in numeric_columns if not col.startswith(("mtd_", "month_reset_"))])
    month_start = f"{year}{month}01"
    month_end = f"{year}{month}31"
    con.execute(
        f"""
        copy (
            with base as (
                {select_sql}
                from read_parquet('{_escape_sql_literal(str(parquet_path))}') as base
            ),
            bars_raw as (
                select
                    substr(trade_date, 1, 4) || '-' || substr(trade_date, 5, 2) || '-' || substr(trade_date, 7, 2) as trade_date,
                    code,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    amount,
                    turnover_rate
                from src.{_quote_identifier(source_table)}
                where trade_date >= '{month_start}' and trade_date <= '{month_end}'
            ),
            bars as (
                select
                    *,
                    close / nullif(lag(close) over (partition by code order by trade_date), 0.0) - 1.0 as month_ret_1d
                from bars_raw
            ),
            extra as (
                select
                    trade_date,
                    code,
                    {extra_select}
                from bars
            )
            select
                base.*,
                {_extra_column_select(variant)}
            from base
            left join extra using (trade_date, code)
            order by trade_date, code
        ) to '{_escape_sql_literal(str(output_path))}' (format parquet, compression zstd, row_group_size 50000)
        """
    )


def _base_select_sql(dataset_version: str, numeric_columns: list[str]) -> str:
    numeric_select = ",\n                    ".join(f"base.{_quote_identifier(col)}" for col in numeric_columns)
    return f"""
                select
                    base.trade_date,
                    base.code,
                    base.feature_set_id,
                    '{dataset_version}' as feature_schema_version,
                    {numeric_select},
                    '{dataset_version}' as dataset_version
            """


def _mtd_select_sql() -> str:
    return """
                    cast(row_number() over (partition by code order by trade_date) as real) as mtd_trading_day_in_month,
                    cast(close / nullif(first_value(close) over (partition by code order by trade_date rows between unbounded preceding and unbounded following), 0.0) - 1.0 as real) as mtd_ret,
                    cast(avg(turnover_rate) over (partition by code order by trade_date rows between unbounded preceding and current row) as real) as mtd_turnover_mean,
                    cast(amount / nullif(avg(amount) over (partition by code order by trade_date rows between unbounded preceding and current row), 0.0) as real) as mtd_amount_ratio,
                    cast(volume / nullif(avg(volume) over (partition by code order by trade_date rows between unbounded preceding and current row), 0.0) as real) as mtd_volume_ratio,
                    cast(close / nullif(max(high) over (partition by code order by trade_date rows between unbounded preceding and 1 preceding), 0.0) - 1.0 as real) as mtd_high_distance,
                    cast(close / nullif(min(low) over (partition by code order by trade_date rows between unbounded preceding and 1 preceding), 0.0) - 1.0 as real) as mtd_low_distance
    """


def _month_reset_select_sql() -> str:
    pieces = [
        "cast(month_ret_1d as real) as month_reset_ret_1d",
        "cast(open / nullif(lag(close) over (partition by code order by trade_date), 0.0) - 1.0 as real) as month_reset_open_gap_pct",
    ]
    for window in WINDOWS:
        frame = f"rows between {window - 1} preceding and current row"
        prev_frame = f"rows between {window} preceding and 1 preceding"
        pieces.extend(
            [
                f"cast(close / nullif(lag(close, {window}) over (partition by code order by trade_date), 0.0) - 1.0 as real) as month_reset_ret_{window}d",
                f"cast(avg(turnover_rate) over (partition by code order by trade_date {frame}) as real) as month_reset_turnover_mean_{window}d",
                f"cast(amount / nullif(avg(amount) over (partition by code order by trade_date {frame}), 0.0) as real) as month_reset_amount_ratio_{window}d",
                f"cast(volume / nullif(avg(volume) over (partition by code order by trade_date {frame}), 0.0) as real) as month_reset_volume_ratio_{window}d",
                f"cast(close / nullif(max(high) over (partition by code order by trade_date {prev_frame}), 0.0) - 1.0 as real) as month_reset_high_distance_{window}d",
                f"cast(close / nullif(min(low) over (partition by code order by trade_date {prev_frame}), 0.0) - 1.0 as real) as month_reset_low_distance_{window}d",
                f"cast(coalesce(stddev_samp(month_ret_1d) over (partition by code order by trade_date {frame}), 0.0) as real) as month_reset_volatility_{window}d",
            ]
        )
    return ",\n                    ".join(pieces)


def _extra_column_select(variant: str) -> str:
    columns = _mtd_columns() if variant == "mtd" else _month_reset_columns()
    return ",\n                ".join(f"extra.{_quote_identifier(col)}" for col in columns)


def _schema_payload(source_schema: dict[str, object], dataset_version: str, numeric_columns: list[str]) -> dict[str, object]:
    payload = {
        "categorical_columns": list(source_schema.get("categorical_columns", [])),
        "dataset_version": dataset_version,
        "excluded_metadata_columns": list(source_schema.get("excluded_metadata_columns", [])),
        "feature_set_id": FEATURE_SET_ID,
        "fill_values": {"numeric": 0.0},
        "numeric_columns": numeric_columns,
        "schema_version": dataset_version,
    }
    payload["schema_hash"] = compute_feature_schema_hash(payload)
    return payload


def _metadata_payload(source_metadata: dict[str, object], dataset_version: str, variant: str, source_db: str) -> dict[str, object]:
    return {
        "compression": "zstd",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "derived_from_dataset_version": source_metadata.get("dataset_version"),
        "derivation": variant,
        "dtype_policy": source_metadata.get("dtype_policy", "float32"),
        "excluded_metadata_columns": list(source_metadata.get("excluded_metadata_columns", [])),
        "feature_set_id": FEATURE_SET_ID,
        "max_date": source_metadata.get("max_date"),
        "min_date": source_metadata.get("min_date"),
        "row_count": source_metadata.get("row_count"),
        "source_db": source_db,
        "source_feature_store": source_metadata.get("source_db"),
        "source_table": source_metadata.get("source_table", "ml_feature_mart_daily"),
        "source_start_date": source_metadata.get("source_start_date"),
        "source_end_date": source_metadata.get("source_end_date"),
    }


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


if __name__ == "__main__":
    main()
