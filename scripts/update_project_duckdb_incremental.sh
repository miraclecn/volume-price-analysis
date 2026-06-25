#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/nan/volume-price-analysis}"
PYTHON_BIN="${PYTHON_BIN:-/home/nan/.miniconda/bin/python}"
SOURCE_DB="${SOURCE_DB:-/home/nan/alpha-data-local/output/research_source.duckdb}"
VPA_DB="${VPA_DB:-outputs/vpa.duckdb}"
ML_CONFIG="${ML_CONFIG:-config/ml_walkforward_adv10m_ret5_alpha_risk.toml}"
ML_DB="${ML_DB:-outputs/ml/ml_ret5_alpha_risk_20260619.duckdb}"
LIVE_DB="${LIVE_DB:-outputs/ml/live_sim/live_sim_state.duckdb}"
ML_FEATURE_SET_ID="${ML_FEATURE_SET_ID:-vpa_d_sequence}"
LIVE_PREDICTION_RUN_ID="${LIVE_PREDICTION_RUN_ID:-wf_v2_ret5_fund_fixed_a160_r120_20260621}"
LIVE_PREDICTION_SCORE_VERSION="${LIVE_PREDICTION_SCORE_VERSION:-v2_alpha_ret5d_fund_fixed_a160_r120_20260621}"
LIVE_PREDICTION_FOLD_ID="${LIVE_PREDICTION_FOLD_ID:-wf_2026}"
VPA_CONFIG="${VPA_CONFIG:-config/default.toml}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/outputs/logs}"
LOCK_FILE="${LOCK_FILE:-${REPO_DIR}/outputs/update_project_duckdb_incremental.lock}"

mkdir -p "$LOG_DIR" "$(dirname "$LOCK_FILE")"
cd "$REPO_DIR"
export SOURCE_DB VPA_DB ML_DB LIVE_DB ML_FEATURE_SET_ID LIVE_PREDICTION_RUN_ID LIVE_PREDICTION_SCORE_VERSION

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
print("ML_FEATURE_MAX=" + max_date(os.environ["ML_DB"], "ml_feature_mart_daily", "trade_date"))
print("ML_TRADEABILITY_MAX=" + max_date(os.environ["ML_DB"], "ml_tradeability_daily", "trade_date"))

prediction_max = ""
if Path(os.environ["ML_DB"]).exists():
    con = duckdb.connect(os.environ["ML_DB"], read_only=True)
    try:
        tables = {
            row[0]
            for row in con.execute(
                "select table_name from information_schema.tables where table_schema = 'main'"
            ).fetchall()
        }
        if "ml_predictions_daily" in tables:
            value = con.execute(
                """
                select max(trade_date)
                from ml_predictions_daily
                where run_id = ?
                  and score_version = ?
                """,
                [os.environ["LIVE_PREDICTION_RUN_ID"], os.environ["LIVE_PREDICTION_SCORE_VERSION"]],
            ).fetchone()[0]
            prediction_max = fmt(value) if value is not None else ""
    finally:
        con.close()
print("ML_PREDICTION_MAX=" + prediction_max)
PY
}

next_day() {
  date -d "$1 + 1 day" +%F
}

lookback_start() {
  date -d "$1 - 420 days" +%F
}

prediction_dates_to_update() {
  "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path
import duckdb

path = os.environ["ML_DB"]
if not Path(path).exists():
    raise SystemExit(0)
con = duckdb.connect(path, read_only=True)
try:
    rows = con.execute(
        """
        select distinct f.trade_date
        from ml_feature_mart_daily f
        left join (
            select distinct trade_date
            from ml_predictions_daily
            where run_id = ?
              and score_version = ?
        ) p
          on f.trade_date = p.trade_date
        where f.feature_set_id = ?
          and f.trade_date <= ?
          and p.trade_date is null
        order by f.trade_date
        """,
        [
            os.environ["LIVE_PREDICTION_RUN_ID"],
            os.environ["LIVE_PREDICTION_SCORE_VERSION"],
            os.environ["ML_FEATURE_SET_ID"],
            os.environ["SOURCE_MAX"],
        ],
    ).fetchall()
finally:
    con.close()
for (value,) in rows:
    print(str(value)[:10])
PY
}

log "starting project DuckDB incremental update"
eval "$(query_dates)"

if [[ -z "${SOURCE_MAX:-}" ]]; then
  log "no source max date found in $SOURCE_DB"
  exit 1
fi

log "source_max=${SOURCE_MAX} vpa_max=${VPA_MAX:-none} ml_db=${ML_DB} ml_feature_max=${ML_FEATURE_MAX:-none} ml_tradeability_max=${ML_TRADEABILITY_MAX:-none} live_prediction_max=${ML_PREDICTION_MAX:-none}"

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
  if [[ -n "${ML_FEATURE_MAX:-}" ]]; then
    ML_START="$(next_day "$ML_FEATURE_MAX")"
  elif [[ -n "${ML_PREDICTION_MAX:-}" && "$ML_PREDICTION_MAX" < "$SOURCE_MAX" ]]; then
    ML_START="$(next_day "$ML_PREDICTION_MAX")"
  elif [[ -n "${ML_TRADEABILITY_MAX:-}" && "$ML_TRADEABILITY_MAX" < "$SOURCE_MAX" ]]; then
    ML_START="$(next_day "$ML_TRADEABILITY_MAX")"
  else
    ML_START="$SOURCE_MAX"
  fi
  ML_BAR_START="$(lookback_start "$ML_START")"
  log "updating ML feature DuckDB ${ML_START}..${SOURCE_MAX}"
  "$PYTHON_BIN" scripts/run_ml_feature_mart.py \
    --config "$ML_CONFIG" \
    --bar-start-date "$ML_BAR_START" \
    --start-date "$ML_START" \
    --end-date "$SOURCE_MAX" \
    --write-duckdb true \
    --write-feature-store false
else
  log "ML feature DuckDB already current"
fi

eval "$(query_dates)"
export SOURCE_MAX
PREDICTION_DATES="$(prediction_dates_to_update)"
if [[ -n "$PREDICTION_DATES" ]]; then
  while IFS= read -r PREDICTION_DATE; do
    [[ -z "$PREDICTION_DATE" ]] && continue
    log "updating profit_protect daily prediction for ${PREDICTION_DATE}"
    "$PYTHON_BIN" scripts/run_profit_protect_daily_prediction.py \
      --ml-db "$ML_DB" \
      --live-db "$LIVE_DB" \
      --as-of-date "$PREDICTION_DATE" \
      --base-feature-set-id "$ML_FEATURE_SET_ID" \
      --run-id "$LIVE_PREDICTION_RUN_ID" \
      --fold-id "$LIVE_PREDICTION_FOLD_ID" \
      --score-version "$LIVE_PREDICTION_SCORE_VERSION"
  done <<< "$PREDICTION_DATES"
else
  log "profit_protect daily predictions already current"
fi

eval "$(query_dates)"
log "finished project DuckDB update: source_max=${SOURCE_MAX} vpa_max=${VPA_MAX:-none} ml_db=${ML_DB} ml_feature_max=${ML_FEATURE_MAX:-none} ml_tradeability_max=${ML_TRADEABILITY_MAX:-none} live_prediction_max=${ML_PREDICTION_MAX:-none}"
