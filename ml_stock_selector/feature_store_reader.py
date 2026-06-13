from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterator

import duckdb
import pandas as pd

from ml_stock_selector.feature_store import compute_feature_schema_hash


@dataclass(frozen=True)
class FeatureStoreSpec:
    feature_store_dir: str
    dataset_version: str
    feature_set_id: str
    schema_version: str | None = None


@dataclass(frozen=True)
class FeatureSchema:
    feature_set_id: str
    dataset_version: str
    schema_version: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    fill_values: dict[str, object]
    excluded_metadata_columns: list[str]
    schema_hash: str | None = None


def load_feature_schema(spec: FeatureStoreSpec) -> FeatureSchema:
    path = _dataset_root(spec) / "feature_schema.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_hash = payload.get("schema_hash")
    if schema_hash and compute_feature_schema_hash(payload) != schema_hash:
        raise ValueError(f"feature schema_hash mismatch: {path}")
    metadata_path = _dataset_root(spec) / "_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_hash = metadata.get("schema_hash")
        if schema_hash and metadata_hash and metadata_hash != schema_hash:
            raise ValueError(f"feature schema_hash mismatch between {path.name} and {metadata_path.name}")
    schema_version = payload.get("schema_version", payload["dataset_version"])
    if spec.schema_version is not None and spec.schema_version != schema_version:
        raise ValueError(f"feature schema_version mismatch: expected {spec.schema_version}, got {schema_version}")
    return FeatureSchema(
        feature_set_id=payload["feature_set_id"],
        dataset_version=payload["dataset_version"],
        schema_version=schema_version,
        numeric_columns=list(payload.get("numeric_columns", [])),
        categorical_columns=list(payload.get("categorical_columns", [])),
        fill_values=dict(payload.get("fill_values", {})),
        excluded_metadata_columns=list(payload.get("excluded_metadata_columns", [])),
        schema_hash=str(schema_hash) if schema_hash else None,
    )


def iter_feature_store_batches(
    spec: FeatureStoreSpec,
    start_date: str,
    end_date: str,
    columns: list[str] | None = None,
    batch_size: int = 50000,
) -> Iterator[pd.DataFrame]:
    schema = load_feature_schema(spec)
    selected = columns or ["trade_date", "code", *schema.numeric_columns]
    selected = _ordered_columns(selected, schema)
    glob = str(_dataset_root(spec) / "year=*" / "month=*" / "*.parquet")
    con = duckdb.connect(":memory:")
    try:
        quoted = ", ".join(_quote_identifier(col) for col in selected)
        cursor = con.execute(
            f"""
            select {quoted}
            from read_parquet(?)
            where trade_date >= ? and trade_date <= ?
            order by trade_date, code
            """,
            [glob, start_date, end_date],
        )
        vectors_per_chunk = max(1, batch_size // 2048)
        while True:
            frame = cursor.fetch_df_chunk(vectors_per_chunk)
            if frame.empty:
                break
            yield frame
    finally:
        con.close()


def _dataset_root(spec: FeatureStoreSpec) -> Path:
    return Path(spec.feature_store_dir) / f"dataset_version={spec.dataset_version}" / f"feature_set_id={spec.feature_set_id}"


def _ordered_columns(columns: list[str], schema: FeatureSchema) -> list[str]:
    requested = set(columns)
    ordered = [col for col in ["trade_date", "code", "feature_set_id", "feature_schema_version"] if col in requested]
    ordered.extend([col for col in schema.numeric_columns if col in requested])
    ordered.extend([col for col in columns if col not in ordered])
    return ordered


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
