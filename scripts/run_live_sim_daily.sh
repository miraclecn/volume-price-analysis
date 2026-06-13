#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/nan/volume-price-analysis"
PYTHON_BIN="${PYTHON_BIN:-/home/nan/.miniconda/bin/python}"
cd "$ROOT_DIR"

mkdir -p outputs/ml/live_sim/logs outputs/ml/live_sim/reports

ARGS=(
  scripts/run_live_sim_daily.py
  --config config/ml_walkforward_adv10m.toml \
  --state-db outputs/ml/live_sim/live_sim_state.duckdb \
  --report-dir outputs/ml/live_sim/reports \
  --account-id preferred_adv10m_paper \
  --initial-cash 300000 \
  --use-feature-store false
)

if [[ -n "${QMT_ORDER_COPY_DIR:-}" ]]; then
  ARGS+=(--qmt-order-copy-dir "$QMT_ORDER_COPY_DIR")
fi

"$PYTHON_BIN" "${ARGS[@]}"
