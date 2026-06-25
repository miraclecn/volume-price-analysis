# Board Execution Data Request

Date: 2026-06-22

Purpose: collect the minimum real execution data needed to decide whether the board-hitting candidate can move from `paper_only` to `live_candidate`.

Current candidate:

```text
strategy line: sealed-board overnight / board hitting research
candidate variant: neutral_expected_pred_ret
ranking: pred_ret * estimated_fill_probability
target attempts: top5 sealed boards per active day
target size: 1% per filled board, max 5% attempted sleeve
current blocker: missing outputs/limit_hit_research/board_execution.duckdb
```

## Required Period

Preferred:

```text
2020-01-01 through 2025-12-31
```

Minimum useful calibration sample:

```text
At least 6 consecutive months with active board attempts, including order/fill logs.
```

If full history is unavailable, prioritize the most recent period with reliable broker fills and order-book snapshots.

## Required Tables

The import schema is fixed by:

```text
sql/create_board_execution_tables.sql
ml_stock_selector/contracts/board_execution_contract.py
```

CSV templates:

```text
outputs/limit_hit_research/board_execution_templates/board_intraday_events_template.csv
outputs/limit_hit_research/board_execution_templates/board_order_book_snapshots_template.csv
outputs/limit_hit_research/board_execution_templates/board_order_fills_template.csv
```

### board_intraday_events

One row per `trade_date/code` board event.

Required columns:

```text
trade_date
code
first_limit_time
last_limit_time
seal_duration_seconds
reopen_count
limit_up
close
is_close_sealed
```

Meaning:

- `first_limit_time`: first time the stock reached limit-up on that trade date.
- `last_limit_time`: last time it sealed at limit-up.
- `seal_duration_seconds`: total sealed duration during the session.
- `reopen_count`: number of times the board reopened after sealing.
- `is_close_sealed`: true if it closed sealed at limit-up.

### board_order_book_snapshots

One or more rows per signal/order time.

Required columns:

```text
trade_date
code
snapshot_time
bid_price_1
bid_volume_1
ask_price_1
ask_volume_1
limit_queue_volume
```

Preferred snapshots:

```text
first_limit_time
signal_time
order_time
close auction / near close
```

The key field is `limit_queue_volume`; without it, queue adverse selection cannot be calibrated.

### board_order_fills

Broker/order log rows.

Required columns:

```text
trade_date
code
signal_time
order_time
side
order_price
order_qty
filled_qty
avg_fill_price
status
```

Rules:

- `side` should be `buy` or `sell`; the current gate uses buy fills.
- `filled_qty > 0` is treated as a real fill.
- Partial fills are valid and should not be dropped.
- Rejected/cancelled orders should be included with `filled_qty = 0` if available.

## Import Command

After replacing template rows with real data:

```bash
python scripts/import_board_execution_data.py \
  --db outputs/limit_hit_research/board_execution.duckdb \
  --events-csv <board_intraday_events.csv> \
  --order-book-csv <board_order_book_snapshots.csv> \
  --fills-csv <board_order_fills.csv> \
  --require-order-book \
  --require-fills \
  --source <vendor_or_broker>
```

## Acceptance Checks

Run the strict contract check:

```bash
python scripts/run_board_execution_contract_check.py \
  --db outputs/limit_hit_research/board_execution.duckdb \
  --require-order-book \
  --require-fills
```

Run the coverage audit:

```bash
python scripts/audit_board_execution_data.py \
  --predictions outputs/limit_hit_research/reports/board_overnight_model_2020_2025/board_overnight_predictions.csv \
  --board-db outputs/limit_hit_research/board_execution.duckdb \
  --out-json outputs/limit_hit_research/reports/board_execution_data_audit.json
```

Run the full downstream refresh:

```bash
python scripts/run_board_research_refresh.py
```

## Promotion Criteria

The board strategy remains blocked unless:

```text
outputs/limit_hit_research/reports/board_candidate_strategy_manifest.json
status = live_candidate
```

Minimum live-candidate criteria:

```text
fillability gate passes
real matched fills/day >= 2
candidate annual return remains positive after real-fill calibration
```

Hard pre-live check:

```bash
python scripts/check_board_live_gate.py \
  --manifest outputs/limit_hit_research/reports/board_candidate_strategy_manifest.json
```

The command must return:

```text
ok = true
exit_code = 0
```

## Current Expected State

Before real data is imported:

```text
audit ok = false
candidate status = paper_only
live gate exit_code = 1
```

This is expected and protects the current `mkt_tier_profit_protect` live sim from unverified board-hitting logic.
