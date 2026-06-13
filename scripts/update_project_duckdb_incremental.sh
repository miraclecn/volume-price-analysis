#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/nan/volume-price-analysis}"
PYTHON_BIN="${PYTHON_BIN:-/home/nan/.miniconda/bin/python}"
SOURCE_DB="${SOURCE_DB:-/home/nan/alpha-data-local/output/research_source.duckdb}"
VPA_DB="${VPA_DB:-outputs/vpa.duckdb}"
ML_CONFIG="${ML_CONFIG:-config/ml_default.toml}"
VPA_CONFIG="${VPA_CONFIG:-config/default.toml}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/outputs/logs}"
LOCK_FILE="${LOCK_FILE:-${REPO_DIR}/outputs/update_project_duckdb_incremental.lock}"

mkdir -p "$LOG_DIR" "$(dirname "$LOCK_FILE")"
cd "$REPO_DIR"
export SOURCE_DB VPA_DB

exec 9>"$LOCK_FILE"
flock -n 9 || {
  echo "$(date -Is) another project DuckDB update is already running"
  exit 0
}

log() {
  echo "$(date -Is) $*"
}

query_dates() {
  "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path
import duckdb


def fmt(value) -> str:
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text[:10]


def max_date(path: str, table: str, column: str) -> str:
    if not Path(path).exists():
        return ""
    con = duckdb.connect(path, read_only=True)
    try:
        tables = {
            row[0]
            for row in con.execute(
                "select table_name from information_schema.tables where table_schema = 'main'"
            ).fetchall()
        }
        if table not in tables:
            return ""
        value = con.execute(f"select max({column}) from {table}").fetchone()[0]
        return fmt(value) if value is not None else ""
    finally:
        con.close()


print("SOURCE_MAX=" + max_date(os.environ["SOURCE_DB"], "stock_bar_normalized_daily", "trade_date"))
print("VPA_MAX=" + max_date(os.environ["VPA_DB"], "vpa_structure_state", "date"))
print("ML_FEATURE_MAX=" + max_date("outputs/ml/ml.duckdb", "ml_feature_mart_daily", "trade_date"))
print("ML_PREDICTION_MAX=" + max_date("outputs/ml/ml.duckdb", "ml_predictions_daily", "trade_date"))
PY
}

next_day() {
  date -d "$1 + 1 day" +%F
}

log "starting project DuckDB incremental update"
eval "$(query_dates)"

if [[ -z "${SOURCE_MAX:-}" ]]; then
  log "no source max date found in $SOURCE_DB"
  exit 1
fi

log "source_max=${SOURCE_MAX} vpa_max=${VPA_MAX:-none} ml_feature_max=${ML_FEATURE_MAX:-none} ml_prediction_max=${ML_PREDICTION_MAX:-none}"

if [[ -z "${VPA_MAX:-}" || "$VPA_MAX" < "$SOURCE_MAX" ]]; then
  VPA_START="${VPA_MAX:-2015-01-05}"
  if [[ -n "${VPA_MAX:-}" ]]; then
    VPA_START="$(next_day "$VPA_MAX")"
  fi
  log "updating VPA DuckDB ${VPA_START}..${SOURCE_MAX}"
  "$PYTHON_BIN" -m vpa_structure_recognizer.batch_runner \
    --config "$VPA_CONFIG" \
    --source "$SOURCE_DB" \
    --output-db "$VPA_DB" \
    --start-date "$VPA_START" \
    --end-date "$SOURCE_MAX"
else
  log "VPA DuckDB already current"
fi

eval "$(query_dates)"
if [[ -z "${ML_FEATURE_MAX:-}" || "$ML_FEATURE_MAX" < "$SOURCE_MAX" ]]; then
  ML_START="${ML_FEATURE_MAX:-2015-01-05}"
  if [[ -n "${ML_FEATURE_MAX:-}" ]]; then
    ML_START="$(next_day "$ML_FEATURE_MAX")"
  fi
  log "updating ML feature DuckDB ${ML_START}..${SOURCE_MAX}"
  "$PYTHON_BIN" scripts/run_ml_feature_mart.py \
    --config "$ML_CONFIG" \
    --start-date "$ML_START" \
    --end-date "$SOURCE_MAX" \
    --write-duckdb true \
    --write-feature-store false
else
  log "ML feature DuckDB already current"
fi

eval "$(query_dates)"
if [[ -z "${ML_PREDICTION_MAX:-}" || "$ML_PREDICTION_MAX" < "$SOURCE_MAX" ]]; then
  log "updating daily signal for ${SOURCE_MAX}"
  "$PYTHON_BIN" scripts/run_ml_daily_signal.py \
    --config "$ML_CONFIG" \
    --as-of-date "$SOURCE_MAX" \
    --use-feature-store false
else
  log "ML daily predictions already current"
fi

eval "$(query_dates)"
log "finished project DuckDB update: source_max=${SOURCE_MAX} vpa_max=${VPA_MAX:-none} ml_feature_max=${ML_FEATURE_MAX:-none} ml_prediction_max=${ML_PREDICTION_MAX:-none}"
