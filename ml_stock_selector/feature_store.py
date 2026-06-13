from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatchcase
import hashlib
from pathlib import Path
import gc
import json
import math
import multiprocessing as mp
from uuid import uuid4

import duckdb
import pandas as pd

DENIED_FEATURE_COLUMNS = {
    "industry_code",
    "industry_name",
    "industry_unknown",
    "industry_missing",
    "exchange",
    "board",
    "is_bse",
    "is_st",
    "is_paused",
    "can_buy_next_open",
    "can_sell_next_open",
}


@dataclass(frozen=True)
class FeatureStoreExportResult:
    root_dir: Path
    feature_schema_path: Path
    metadata_path: Path
    row_count: int
    numeric_columns: list[str]


@dataclass(frozen=True)
class FeatureAllowlist:
    feature_set_id: str
    schema_version: str
    include_patterns: list[str]
    exclude_columns: list[str]

    def allows(self, column: str) -> bool:
        if column in DENIED_FEATURE_COLUMNS or column in set(self.exclude_columns):
            return False
        return not self.include_patterns or any(fnmatchcase(column, pattern) for pattern in self.include_patterns)


def export_feature_store(
    con,
    output_dir: Path | str,
    dataset_version: str,
    feature_set_id: str,
    start_date: str,
    end_date: str,
    *,
    chunk_size: int = 20000,
    row_group_size: int = 50000,
    compression: str = "zstd",
    schema_sample_size: int = 100000,
    isolate_month_exports: bool = False,
    allowlist_path: Path | str | None = None,
) -> FeatureStoreExportResult:
    root_dir = Path(output_dir) / f"dataset_version={dataset_version}" / f"feature_set_id={feature_set_id}"
    root_dir.mkdir(parents=True, exist_ok=True)
    allowlist = load_feature_allowlist(allowlist_path) if allowlist_path is not None else None
    row_count, min_date, max_date = _source_stats(con, feature_set_id, start_date, end_date)
    numeric_columns = _infer_numeric_columns(con, feature_set_id, start_date, end_date, schema_sample_size, allowlist)

    source_db_path = _connection_database_path(con) if isolate_month_exports else None
    for year_month in _source_months(con, feature_set_id, start_date, end_date):
        year, month = [int(part) for part in year_month.split("-")]
        partition_dir = root_dir / f"year={year:04d}" / f"month={month:02d}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        if source_db_path:
            _copy_month_in_subprocess(
                source_db_path,
                partition_dir / "part-00000.parquet",
                feature_set_id,
                max(start_date, f"{year_month}-01"),
                min(end_date, _month_end(year_month)),
                dataset_version,
                numeric_columns,
                compression,
                row_group_size,
            )
            continue
        month_con = con
        month_con.execute("set preserve_insertion_order = false")
        month_con.execute("set threads = 1")
        _copy_month_to_parquet(
            month_con,
            partition_dir / "part-00000.parquet",
            feature_set_id,
            max(start_date, f"{year_month}-01"),
            min(end_date, _month_end(year_month)),
            dataset_version,
            numeric_columns,
            compression=compression,
            row_group_size=row_group_size,
        )

    schema_path = root_dir / "feature_schema.json"
    metadata_path = root_dir / "_metadata.json"
    schema_payload = _feature_schema_payload(
        feature_set_id,
        dataset_version,
        numeric_columns,
        allowlist=allowlist,
    )
    _write_json(schema_path, schema_payload)
    _write_json(
        metadata_path,
        _feature_store_metadata_payload(
            dataset_version=dataset_version,
            feature_set_id=feature_set_id,
            source_table="ml_feature_mart_daily",
            source_db=_connection_database_path(con) or ":memory:",
            source_start_date=start_date,
            source_end_date=end_date,
            min_date=min_date,
            max_date=max_date,
            row_count=row_count,
            compression=compression,
            schema_hash=str(schema_payload["schema_hash"]),
        ),
    )
    return FeatureStoreExportResult(root_dir, schema_path, metadata_path, row_count, numeric_columns)


def write_feature_frame_to_feature_store(
    feature_mart: pd.DataFrame,
    output_dir: Path | str,
    dataset_version: str,
    feature_set_id: str,
    *,
    source_db: str,
    compression: str = "zstd",
    row_group_size: int = 50000,
    allowlist_path: Path | str | None = None,
) -> FeatureStoreExportResult:
    allowlist = load_feature_allowlist(allowlist_path) if allowlist_path is not None else None
    source = feature_mart[feature_mart["feature_set_id"].astype(str) == feature_set_id].copy()
    if source.empty:
        min_date = max_date = None
    else:
        min_date = str(source["trade_date"].min())
        max_date = str(source["trade_date"].max())
    numeric_columns = _infer_numeric_columns_from_series(source.get("features_json", pd.Series(dtype=object)), allowlist)
    root_dir = Path(output_dir) / f"dataset_version={dataset_version}" / f"feature_set_id={feature_set_id}"
    root_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    try:
        for year_month, chunk in source.groupby(source["trade_date"].astype(str).str.slice(0, 7), sort=True):
            year, month = [int(part) for part in str(year_month).split("-")]
            partition_dir = root_dir / f"year={year:04d}" / f"month={month:02d}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            out = _chunk_to_feature_frame(chunk.reset_index(drop=True), dataset_version, numeric_columns)
            out = out.drop(columns=["year", "month"], errors="ignore")
            _copy_dataframe_to_parquet(
                con,
                out,
                partition_dir / "part-00000.parquet",
                compression=compression,
                row_group_size=row_group_size,
            )
    finally:
        con.close()
    schema_payload = _feature_schema_payload(feature_set_id, dataset_version, numeric_columns, allowlist=allowlist)
    schema_path = root_dir / "feature_schema.json"
    metadata_path = root_dir / "_metadata.json"
    _write_json(schema_path, schema_payload)
    _write_json(
        metadata_path,
        _feature_store_metadata_payload(
            dataset_version=dataset_version,
            feature_set_id=feature_set_id,
            source_table="feature_mart_dataframe",
            source_db=source_db,
            source_start_date=min_date,
            source_end_date=max_date,
            min_date=min_date,
            max_date=max_date,
            row_count=len(source),
            compression=compression,
            schema_hash=str(schema_payload["schema_hash"]),
        ),
    )
    return FeatureStoreExportResult(root_dir, schema_path, metadata_path, len(source), numeric_columns)


def _copy_month_in_subprocess(
    source_db_path: str,
    parquet_path: Path,
    feature_set_id: str,
    start_date: str,
    end_date: str,
    dataset_version: str,
    numeric_columns: list[str],
    compression: str,
    row_group_size: int,
) -> None:
    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=_copy_month_worker,
        args=(
            source_db_path,
            str(parquet_path),
            feature_set_id,
            start_date,
            end_date,
            dataset_version,
            numeric_columns,
            compression,
            row_group_size,
        ),
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"month parquet export failed for {start_date}..{end_date} with exit code {process.exitcode}")


def _copy_month_worker(
    source_db_path: str,
    parquet_path: str,
    feature_set_id: str,
    start_date: str,
    end_date: str,
    dataset_version: str,
    numeric_columns: list[str],
    compression: str,
    row_group_size: int,
) -> None:
    month_con = duckdb.connect(source_db_path, read_only=True)
    try:
        month_con.execute("set preserve_insertion_order = false")
        month_con.execute("set threads = 1")
        _copy_month_to_parquet(
            month_con,
            Path(parquet_path),
            feature_set_id,
            start_date,
            end_date,
            dataset_version,
            numeric_columns,
            compression=compression,
            row_group_size=row_group_size,
        )
    finally:
        month_con.close()


def parse_numeric_features(
    features_json: object,
    numeric_columns: list[str] | None = None,
    allowlist: FeatureAllowlist | None = None,
) -> dict[str, float]:
    raw = json.loads(features_json or "{}") if isinstance(features_json, str) else dict(features_json or {})
    values: dict[str, float] = {}
    allowed = set(numeric_columns) if numeric_columns is not None else None
    for key, value in raw.items():
        if key in DENIED_FEATURE_COLUMNS or (allowed is not None and key not in allowed):
            continue
        if allowlist is not None and not allowlist.allows(key):
            continue
        if isinstance(value, bool):
            values[key] = float(int(value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            values[key] = float(value)
    if numeric_columns is not None:
        for col in numeric_columns:
            values.setdefault(col, 0.0)
    return values


def load_feature_allowlist(path: Path | str) -> FeatureAllowlist:
    data: dict[str, object] = {"include_patterns": [], "exclude_columns": []}
    current_list: str | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-") and current_list is not None:
            data[current_list].append(_clean_yaml_scalar(line[1:].strip()))  # type: ignore[index]
            continue
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if value == "":
            current_list = key
            data.setdefault(key, [])
        else:
            current_list = None
            data[key] = _clean_yaml_scalar(value)
    return FeatureAllowlist(
        feature_set_id=str(data.get("feature_set_id", "")),
        schema_version=str(data.get("schema_version", "")),
        include_patterns=list(data.get("include_patterns", [])),
        exclude_columns=list(data.get("exclude_columns", [])),
    )


def compute_feature_schema_hash(payload: dict[str, object]) -> str:
    clean = {key: value for key, value in payload.items() if key != "schema_hash"}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _clean_yaml_scalar(value: str) -> str:
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def _infer_numeric_columns(
    con,
    feature_set_id: str,
    start_date: str,
    end_date: str,
    sample_size: int,
    allowlist: FeatureAllowlist | None,
) -> list[str]:
    columns: set[str] = set()
    sample = con.execute(
        """
        select features_json
        from ml_feature_mart_daily
        where feature_set_id = ?
          and trade_date >= ?
          and trade_date <= ?
        limit ?
        """,
        [feature_set_id, start_date, end_date, sample_size],
    ).fetchdf()
    columns.update(_infer_numeric_columns_from_series(sample["features_json"], allowlist))
    del sample
    gc.collect()
    return sorted(columns)


def _infer_numeric_columns_from_series(series: pd.Series, allowlist: FeatureAllowlist | None) -> list[str]:
    columns: set[str] = set()
    for value in series:
        columns.update(parse_numeric_features(value, allowlist=allowlist).keys())
    return sorted(columns)


def _chunk_to_feature_frame(chunk: pd.DataFrame, dataset_version: str, numeric_columns: list[str]) -> pd.DataFrame:
    feature_rows = [parse_numeric_features(value, numeric_columns) for value in chunk["features_json"]]
    features = pd.DataFrame(feature_rows, columns=numeric_columns).fillna(0.0).astype("float32")
    keys = chunk[["trade_date", "code", "feature_set_id"]].reset_index(drop=True).copy()
    keys["feature_schema_version"] = dataset_version
    dates = pd.to_datetime(keys["trade_date"])
    partitions = pd.DataFrame({"year": dates.dt.year, "month": dates.dt.month})
    return pd.concat([keys, features.reset_index(drop=True), partitions], axis=1)


def _fetch_feature_chunk(
    con,
    feature_set_id: str,
    start_date: str,
    end_date: str,
    limit: int,
    offset: int,
) -> pd.DataFrame:
    return con.execute(
        """
        select trade_date, code, feature_set_id, features_json
        from ml_feature_mart_daily
        where feature_set_id = ?
          and trade_date >= ?
          and trade_date <= ?
        limit ? offset ?
        """,
        [feature_set_id, start_date, end_date, limit, offset],
    ).fetchdf()


def _iter_source_chunks(con, feature_set_id: str, start_date: str, end_date: str, chunk_size: int):
    months = con.execute(
        """
        select distinct substr(trade_date, 1, 7) as year_month
        from ml_feature_mart_daily
        where feature_set_id = ? and trade_date >= ? and trade_date <= ?
        order by year_month
        """,
        [feature_set_id, start_date, end_date],
    ).fetchall()
    for (year_month,) in months:
        month_start = max(start_date, f"{year_month}-01")
        month_end = min(end_date, _month_end(year_month))
        offset = 0
        while True:
            chunk = _fetch_feature_chunk_for_range(con, feature_set_id, month_start, month_end, chunk_size, offset)
            if chunk.empty:
                break
            yield chunk
            offset += len(chunk)


def _fetch_feature_chunk_for_range(
    con,
    feature_set_id: str,
    start_date: str,
    end_date: str,
    limit: int,
    offset: int,
) -> pd.DataFrame:
    return con.execute(
        """
        select trade_date, code, feature_set_id, features_json
        from ml_feature_mart_daily
        where feature_set_id = ?
          and trade_date >= ?
          and trade_date <= ?
        limit ? offset ?
        """,
        [feature_set_id, start_date, end_date, limit, offset],
    ).fetchdf()


def _month_end(year_month: str) -> str:
    year, month = [int(part) for part in year_month.split("-")]
    if month == 12:
        return f"{year}-12-31"
    import calendar

    return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"


def _source_stats(con, feature_set_id: str, start_date: str, end_date: str) -> tuple[int, str | None, str | None]:
    row = con.execute(
        """
        select count(*), min(trade_date), max(trade_date)
        from ml_feature_mart_daily
        where feature_set_id = ? and trade_date >= ? and trade_date <= ?
        """,
        [feature_set_id, start_date, end_date],
    ).fetchone()
    return int(row[0]), row[1], row[2]


def _source_row_count(con, feature_set_id: str, start_date: str, end_date: str) -> int:
    return _source_stats(con, feature_set_id, start_date, end_date)[0]


def _source_months(con, feature_set_id: str, start_date: str, end_date: str) -> list[str]:
    return [
        row[0]
        for row in con.execute(
            """
            select distinct substr(trade_date, 1, 7) as year_month
            from ml_feature_mart_daily
            where feature_set_id = ? and trade_date >= ? and trade_date <= ?
            order by year_month
            """,
            [feature_set_id, start_date, end_date],
        ).fetchall()
    ]


def _source_years(con, feature_set_id: str, start_date: str, end_date: str) -> list[str]:
    return [
        row[0]
        for row in con.execute(
            """
            select distinct substr(trade_date, 1, 4) as year
            from ml_feature_mart_daily
            where feature_set_id = ? and trade_date >= ? and trade_date <= ?
            order by year
            """,
            [feature_set_id, start_date, end_date],
        ).fetchall()
    ]


def _connection_database_path(con) -> str | None:
    rows = con.execute("pragma database_list").fetchall()
    for _, name, path in rows:
        if name != "temp" and path:
            return str(path)
    return None


def _copy_month_to_parquet(
    con,
    path: Path,
    feature_set_id: str,
    start_date: str,
    end_date: str,
    dataset_version: str,
    numeric_columns: list[str],
    *,
    compression: str,
    row_group_size: int,
) -> None:
    feature_sql = ",\n               ".join(
        f"cast(coalesce(try_cast(json_extract(features_json, '{_json_path(col)}') as double), 0.0) as real) as {_quote_identifier(col)}"
        for col in numeric_columns
    )
    select_features = f",\n               {feature_sql}" if feature_sql else ""
    sql = f"""
        copy (
            select trade_date,
                   code,
                   feature_set_id,
                   {_quote_literal(dataset_version)} as feature_schema_version
                   {select_features}
            from ml_feature_mart_daily
            where feature_set_id = {_quote_literal(feature_set_id)}
              and trade_date >= {_quote_literal(start_date)}
              and trade_date <= {_quote_literal(end_date)}
        ) to {_quote_literal(str(path))} (format parquet, compression {_quote_literal(compression)}, row_group_size {int(row_group_size)})
        """
    con.execute(sql)


def _copy_dataset_to_partitioned_parquet(
    con,
    root_dir: Path,
    feature_set_id: str,
    start_date: str,
    end_date: str,
    dataset_version: str,
    numeric_columns: list[str],
    *,
    compression: str,
    row_group_size: int,
) -> None:
    feature_sql = ",\n                   ".join(
        f"cast(coalesce(try_cast(json_extract(features_json, '{_json_path(col)}') as double), 0.0) as real) as {_quote_identifier(col)}"
        for col in numeric_columns
    )
    select_features = f",\n                   {feature_sql}" if feature_sql else ""
    sql = f"""
        copy (
            select trade_date,
                   code,
                   feature_set_id,
                   {_quote_literal(dataset_version)} as feature_schema_version
                   {select_features},
                   substr(trade_date, 1, 4) as year,
                   substr(trade_date, 6, 2) as month
            from ml_feature_mart_daily
            where feature_set_id = {_quote_literal(feature_set_id)}
              and trade_date >= {_quote_literal(start_date)}
              and trade_date <= {_quote_literal(end_date)}
        ) to {_quote_literal(str(root_dir))}
        (format parquet, compression {_quote_literal(compression)}, row_group_size {int(row_group_size)}, partition_by (year, month), filename_pattern 'part-{{i}}')
        """
    con.execute(sql)


def _copy_year_to_partitioned_parquet(
    con,
    root_dir: Path,
    feature_set_id: str,
    start_date: str,
    end_date: str,
    dataset_version: str,
    numeric_columns: list[str],
    *,
    compression: str,
    row_group_size: int,
) -> None:
    feature_sql = ",\n                   ".join(
        f"cast(coalesce(try_cast(json_extract(features_json, '{_json_path(col)}') as double), 0.0) as real) as {_quote_identifier(col)}"
        for col in numeric_columns
    )
    select_features = f",\n                   {feature_sql}" if feature_sql else ""
    sql = f"""
        copy (
            select trade_date,
                   code,
                   feature_set_id,
                   {_quote_literal(dataset_version)} as feature_schema_version
                   {select_features},
                   substr(trade_date, 1, 4) as year,
                   substr(trade_date, 6, 2) as month
            from ml_feature_mart_daily
            where feature_set_id = {_quote_literal(feature_set_id)}
              and trade_date >= {_quote_literal(start_date)}
              and trade_date <= {_quote_literal(end_date)}
        ) to {_quote_literal(str(root_dir))}
        (format parquet, compression {_quote_literal(compression)}, row_group_size {int(row_group_size)}, partition_by (year, month), filename_pattern 'part-{{i}}')
        """
    con.execute(sql)


def _copy_dataframe_to_parquet(con, frame: pd.DataFrame, path: Path, *, compression: str, row_group_size: int) -> None:
    temp_name = f"_feature_store_{uuid4().hex}"
    con.register(temp_name, frame)
    try:
        con.execute(
            f"copy {temp_name} to ? (format parquet, compression ?, row_group_size ?)",
            [str(path), compression, row_group_size],
        )
    finally:
        con.unregister(temp_name)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _feature_schema_payload(
    feature_set_id: str,
    dataset_version: str,
    numeric_columns: list[str],
    *,
    allowlist: FeatureAllowlist | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "feature_set_id": feature_set_id,
        "dataset_version": dataset_version,
        "schema_version": allowlist.schema_version if allowlist and allowlist.schema_version else dataset_version,
        "numeric_columns": numeric_columns,
        "categorical_columns": [],
        "fill_values": {"numeric": 0.0},
        "excluded_metadata_columns": sorted(DENIED_FEATURE_COLUMNS | set(allowlist.exclude_columns if allowlist else [])),
    }
    if allowlist is not None:
        payload["allowlist"] = {
            "include_patterns": allowlist.include_patterns,
            "exclude_columns": allowlist.exclude_columns,
        }
    payload["schema_hash"] = compute_feature_schema_hash(payload)
    return payload


def _feature_store_metadata_payload(
    *,
    dataset_version: str,
    feature_set_id: str,
    source_table: str,
    source_db: str,
    source_start_date: str | None,
    source_end_date: str | None,
    min_date: str | None,
    max_date: str | None,
    row_count: int,
    compression: str,
    schema_hash: str,
) -> dict[str, object]:
    return {
        "dataset_version": dataset_version,
        "feature_set_id": feature_set_id,
        "source_table": source_table,
        "source_db": source_db,
        "source_start_date": source_start_date,
        "source_end_date": source_end_date,
        "min_date": min_date,
        "max_date": max_date,
        "row_count": row_count,
        "dtype_policy": "float32",
        "compression": compression,
        "schema_hash": schema_hash,
        "excluded_metadata_columns": sorted(DENIED_FEATURE_COLUMNS),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _json_path(value: str) -> str:
    return '$."' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
