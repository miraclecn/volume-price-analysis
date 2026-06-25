#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/nan/volume-price-analysis"
PYTHON_BIN="${PYTHON_BIN:-/home/nan/.miniconda/bin/python}"
LOCK_FILE="${LOCK_FILE:-${ROOT_DIR}/outputs/update_project_duckdb_incremental.lock}"
LIVE_SIM_LOCK_TIMEOUT_SECONDS="${LIVE_SIM_LOCK_TIMEOUT_SECONDS:-1800}"
cd "$ROOT_DIR"

mkdir -p outputs/ml/live_sim/logs outputs/ml/live_sim/reports "$(dirname "$LOCK_FILE")"

exec 9>"$LOCK_FILE"
flock -w "$LIVE_SIM_LOCK_TIMEOUT_SECONDS" 9 || {
  echo "$(date -Is) live sim could not acquire update lock within ${LIVE_SIM_LOCK_TIMEOUT_SECONDS}s"
  exit 1
}

ARGS=(
  scripts/run_live_sim_daily.py
  --config config/ml_walkforward_adv10m_ret5_alpha_risk.toml \
  --state-db outputs/ml/live_sim/live_sim_state.duckdb \
  --report-dir outputs/ml/live_sim/reports \
  --account-id profit_protect_paper \
  --initial-cash 1000000 \
  --prediction-source auto
)

if [[ -n "${QMT_ORDER_COPY_DIR:-}" ]]; then
  ARGS+=(--qmt-order-copy-dir "$QMT_ORDER_COPY_DIR")
fi

"$PYTHON_BIN" "${ARGS[@]}"
