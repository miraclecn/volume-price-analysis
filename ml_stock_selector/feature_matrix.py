from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import pandas as pd

from ml_stock_selector.constants import FEATURE_SCHEMA_V2_NO_INDUSTRY, MISSING_CATEGORY, UNKNOWN_CATEGORY


@dataclass(frozen=True)
class FeatureSchema:
    feature_set_id: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    output_columns: list[str]
    category_levels: dict[str, list[str]]
    fill_values: dict[str, object]
    schema_version: str = "v1"


def build_feature_matrix(
    feature_mart_or_samples: pd.DataFrame,
    feature_set_id: str,
    schema: FeatureSchema | None = None,
    fit: bool = False,
    deny_industry: bool = False,
) -> tuple[pd.DataFrame, FeatureSchema]:
    raw = _expand_json(feature_mart_or_samples["features_json"])
    if deny_industry or (schema is not None and schema.schema_version == FEATURE_SCHEMA_V2_NO_INDUSTRY):
        raw = _drop_industry_features(raw)
    if fit:
        schema = _fit_schema(raw, feature_set_id, schema_version=FEATURE_SCHEMA_V2_NO_INDUSTRY if deny_industry else "v1")
    if schema is None:
        raise ValueError("schema is required when fit=False")
    matrix = pd.DataFrame(index=raw.index)
    for col in schema.numeric_columns:
        matrix[col] = pd.to_numeric(raw.get(col, 0.0), errors="coerce").fillna(0.0)
    for col in schema.categorical_columns:
        values = raw.get(col, pd.Series([None] * len(raw), index=raw.index)).fillna(MISSING_CATEGORY).astype(str)
        values = values.where(values.isin(schema.category_levels[col]), UNKNOWN_CATEGORY)
        for level in schema.category_levels[col]:
            matrix[f"{col}={level}"] = (values == level).astype(float)
    for col in schema.output_columns:
        if col not in matrix:
            matrix[col] = 0.0
    return matrix[schema.output_columns].astype(float), schema


def save_feature_schema(schema: FeatureSchema, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(schema), indent=2, sort_keys=True), encoding="utf-8")


def load_feature_schema(path: Path | str) -> FeatureSchema:
    return FeatureSchema(**json.loads(Path(path).read_text(encoding="utf-8")))


def _expand_json(series: pd.Series) -> pd.DataFrame:
    rows = [json.loads(value or "{}") for value in series]
    return pd.DataFrame(rows, index=series.index)


def _fit_schema(raw: pd.DataFrame, feature_set_id: str, schema_version: str = "v1") -> FeatureSchema:
    numeric_columns = []
    categorical_columns = []
    for col in raw.columns:
        values = raw[col].dropna()
        if values.empty:
            numeric_columns.append(col)
        elif all(isinstance(value, (int, float, bool)) for value in values):
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)
    category_levels = {}
    output_columns = sorted(numeric_columns)
    for col in sorted(categorical_columns):
        levels = sorted({str(value) for value in raw[col].dropna()})
        levels = [MISSING_CATEGORY, UNKNOWN_CATEGORY] + [level for level in levels if level not in {MISSING_CATEGORY, UNKNOWN_CATEGORY}]
        category_levels[col] = levels
        output_columns.extend([f"{col}={level}" for level in levels])
    return FeatureSchema(
        feature_set_id=feature_set_id,
        numeric_columns=sorted(numeric_columns),
        categorical_columns=sorted(categorical_columns),
        output_columns=output_columns,
        category_levels=category_levels,
        fill_values={col: 0.0 for col in numeric_columns},
        schema_version=schema_version,
    )


def _drop_industry_features(raw: pd.DataFrame) -> pd.DataFrame:
    denied = {
        col
        for col in raw.columns
        if col in {"industry_code", "industry_name", "industry_unknown"} or col.startswith("industry_")
    }
    return raw.drop(columns=sorted(denied), errors="ignore")
