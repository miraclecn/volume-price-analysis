# VPA Structure Recognizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the A-share multi-level volume-price structure recognizer in this repository, while treating upstream raw data preparation as an external dependency.

**Architecture:** This project reads prepared daily bars, tradeability, industry, index, and market breadth data through read-only adapters, then produces project-owned `vpa_*` derived tables, ratings, validation metrics, and Excel/report outputs. Raw data collection, corporate action adjustment, PIT reference construction, and upstream data quality repair stay outside this repo.

**Tech Stack:** Python, DuckDB Python API, pandas/openpyxl for Excel output, pytest for tests. Do not add new dependencies without explicit approval; prefer stdlib TOML config over YAML parsing unless a YAML parser is approved.

---

## Scope Boundary

In scope for this repository:

- Read-only adapters for external DuckDB data sources.
- Project-local derived data: `vpa_features`, `vpa_trend_context`, `vpa_bar_context_labels`, `vpa_sequence_stats`, `vpa_structure_state`, reports, and validation tables.
- Feature engineering, multi-window labels, sequence patterns, state classification, top-down ranking, Excel export, and stock report formatting.

Out of scope for this repository:

- Raw data download, qfq adjustment, PIT reference construction, trading calendar maintenance, ST/suspension/limit repair, and permanent upstream stock/sector/market marts.

Observed temporary data sources:

- `/home/nan/alpha-find-v2/output/research_source.duckdb`
  - `daily_bar_pit`: 11,715,671 rows, 20140102..20260522, 5,766 securities.
  - `index_daily_bar_pit`: 9,030 rows, 20140102..20260525, 3 indexes.
  - `industry_classification_pit`, `tradeability_state_daily`, `market_trade_calendar`.
- `/home/nan/alpha-find/output/stock_data_audited.duckdb`
  - `mart_kline_qfq`: 16,908,152 rows, 19901219..20260522, 5,846 codes.

Prefer `alpha-find-v2/output/research_source.duckdb` for PIT-aware integration. Use `alpha-find/output/stock_data_audited.duckdb` only as a fallback or history-extension source.

## Target File Structure

- Create: `pyproject.toml`
- Create: `config/default.toml`
- Create: `vpa_structure_recognizer/__init__.py`
- Create: `vpa_structure_recognizer/config.py`
- Create: `vpa_structure_recognizer/models.py`
- Create: `vpa_structure_recognizer/data_sources.py`
- Create: `vpa_structure_recognizer/storage.py`
- Create: `vpa_structure_recognizer/feature_engineering.py`
- Create: `vpa_structure_recognizer/trend_context.py`
- Create: `vpa_structure_recognizer/bar_labeler.py`
- Create: `vpa_structure_recognizer/sequence_analyzer.py`
- Create: `vpa_structure_recognizer/state_classifier.py`
- Create: `vpa_structure_recognizer/top_down_ranker.py`
- Create: `vpa_structure_recognizer/backtest_validator.py`
- Create: `vpa_structure_recognizer/excel_exporter.py`
- Create: `vpa_structure_recognizer/report_formatter.py`
- Create: `vpa_structure_recognizer/pipeline.py`
- Create: `scripts/run_vpa_structure.py`
- Create: `sql/create_vpa_tables.sql`
- Create: `tests/fixtures/`
- Create: `tests/test_*.py` files listed below.

## Acceptance Criteria

- Same scope/date can produce separate labels for windows `10, 20, 30, 60, 120, 240`.
- All price movement, body, shadow, range, and return features use percentages.
- Volume strength is computed from the active `window_n`.
- Single-day labels never emit accumulation/distribution conclusions.
- Trend context uses parent windows.
- Stage/state classification uses multi-day sequence and trend/position context.
- Ranking runs top-down: all-A market, sector, then stock.
- Excel output includes market, sector, stock, multi-window labels, and validation sheets.
- Historical batch run can target 2022, 2023, and 2024 date ranges.

## Task 1: Scaffold Package, Config, and Test Harness

**Files:**
- Create: `pyproject.toml`
- Create: `config/default.toml`
- Create: `vpa_structure_recognizer/config.py`
- Test: `tests/test_config.py`

- [ ] Define project metadata and pytest configuration in `pyproject.toml`.
- [ ] Add `config/default.toml` with windows, parent-window mapping, volume thresholds, label thresholds, scoring weights, output paths, and external DuckDB paths.
- [ ] Implement a stdlib TOML loader in `config.py` with typed accessors for windows and thresholds.
- [ ] Test that default windows equal `[10, 20, 30, 60, 120, 240]`, parent mapping matches the spec, and scoring weights sum to `1.0`.
- [ ] Run: `python -m pytest tests/test_config.py -v`.

## Task 2: Define Data Contracts and Read-Only DuckDB Adapters

**Files:**
- Create: `vpa_structure_recognizer/models.py`
- Create: `vpa_structure_recognizer/data_sources.py`
- Test: `tests/test_data_sources.py`

- [ ] Define normalized contracts for stock, sector, and market bars with spec-required columns.
- [ ] Implement `ResearchSourceDuckDB` read-only queries for:
  - stock bars from `daily_bar_pit`, using adjusted OHLC fields and `lag(close_adj)` for adjusted `prev_close`.
  - tradeability from `tradeability_state_daily`.
  - industry classification from `industry_classification_pit`.
  - index bars from `index_daily_bar_pit`.
- [ ] Implement fallback `AuditedStockDuckDB` for `mart_kline_qfq` when PIT source is unavailable.
- [ ] Test adapters against small synthetic DuckDB fixtures, not the full external databases.
- [ ] Run: `python -m pytest tests/test_data_sources.py -v`.

## Task 3: Create Project-Owned VPA Storage

**Files:**
- Create: `sql/create_vpa_tables.sql`
- Create: `vpa_structure_recognizer/storage.py`
- Test: `tests/test_storage_schema.py`

- [ ] Create DuckDB tables matching spec section 6: `vpa_features`, `vpa_trend_context`, `vpa_bar_context_labels`, `vpa_sequence_stats`, and `vpa_structure_state`.
- [ ] Use primary-key-equivalent uniqueness checks on `(date, scope_type, scope_id, window_n)` where the spec requires it.
- [ ] Implement `init_vpa_db(path)` and append/upsert helpers in `storage.py`.
- [ ] Test schema creation and duplicate-key handling in an in-memory DuckDB database.
- [ ] Run: `python -m pytest tests/test_storage_schema.py -v`.

## Task 4: Build Optional Project-Local Aggregates

**Files:**
- Create: `vpa_structure_recognizer/market_aggregates.py`
- Test: `tests/test_market_aggregates.py`

- [ ] Implement all-A equal-weight OHLC and breadth aggregation from normalized stock bars when an upstream market table is not available.
- [ ] Implement sector equal-weight aggregation by `industry_code` only as a project-local derived fallback.
- [ ] Keep aggregate output under project-owned `vpa_*` storage or `outputs/intermediate/`; do not write back to upstream DuckDB files.
- [ ] Test advancer/decliner counts, limit counts, median return, new-high/new-low counts, and aggregate amount/volume.
- [ ] Run: `python -m pytest tests/test_market_aggregates.py -v`.

## Task 5: Implement Percentage Feature Engineering

**Files:**
- Create: `vpa_structure_recognizer/feature_engineering.py`
- Test: `tests/test_features.py`

- [ ] Compute `ret_pct`, `range_pct`, `body_pct`, `upper_shadow_pct`, and `lower_shadow_pct` using `prev_close`.
- [ ] Compute K-line structure ratios: `body_ratio`, `upper_shadow_ratio`, `lower_shadow_ratio`, and `close_position`.
- [ ] For each window, compute `vol_ma_n`, `vol_rvol_n`, `range_pct_ma_n`, `range_rvol_n`, `body_pct_ma_n`, `body_rvol_n`, `price_high_n`, `price_low_n`, `price_position_n`, `ma_n`, and `ma_slope_n`.
- [ ] Handle zero ranges and missing previous close deterministically by returning null-safe values and `UNKNOWN` downstream context.
- [ ] Run: `python -m pytest tests/test_features.py -v`.

## Task 6: Implement Parent-Window Trend Context

**Files:**
- Create: `vpa_structure_recognizer/trend_context.py`
- Test: `tests/test_trend_context.py`

- [ ] Map each `window_n` to its parent window using config.
- [ ] Compute parent high/low, parent price position, parent MA, parent MA slope, trend label, position label, and trend strength score.
- [ ] Use labels `UPTREND`, `DOWNTREND`, `SIDEWAYS`, `RECOVERING`, `WEAKENING`, and `UNKNOWN`.
- [ ] Test boundary values for `LOW`, `MID_LOW`, `MID`, `MID_HIGH`, and `HIGH`.
- [ ] Run: `python -m pytest tests/test_trend_context.py -v`.

## Task 7: Implement Single-Day Bar Labels

**Files:**
- Create: `vpa_structure_recognizer/bar_labeler.py`
- Test: `tests/test_bar_labels.py`

- [ ] Implement volume level classification: `LOW_VOLUME`, `NORMAL_VOLUME`, `MILD_HIGH_VOLUME`, `HIGH_VOLUME`, and `EXTREME_HIGH_VOLUME`.
- [ ] Implement normal labels: `NORMAL_UP_CONFIRM`, `NORMAL_DOWN_CONFIRM`, and `LOW_VOLUME_SMALL_MOVE`.
- [ ] Implement abnormal labels: `HIGH_VOLUME_LOW_PROGRESS`, `HIGH_VOLUME_UPPER_SUPPLY`, `HIGH_VOLUME_LOWER_SUPPORT`, `LOW_VOLUME_BIG_UP`, `LOW_VOLUME_BIG_DOWN`, `BREAKOUT_PULLBACK`, and `BREAKDOWN_RECOVERY`.
- [ ] Compute `bull_bear_score`, `supply_score`, `demand_score`, and `volatility_score` from label inputs.
- [ ] Test that no single-day label emits `POSSIBLE_ACCUMULATION` or `POSSIBLE_DISTRIBUTION`.
- [ ] Run: `python -m pytest tests/test_bar_labels.py -v`.

## Task 8: Implement Multi-Day Sequence Analysis

**Files:**
- Create: `vpa_structure_recognizer/sequence_analyzer.py`
- Test: `tests/test_sequence_patterns.py`

- [ ] Compute counts and ratios required by `vpa_sequence_stats`.
- [ ] Split each window into `previous_part` and `last_part` halves and compute `bull_score_change`.
- [ ] Implement sequence patterns: `DECLINE_EXHAUSTION_PATTERN`, `LOW_LEVEL_SUPPORT_PATTERN`, `HEALTHY_UPTREND_PATTERN`, `HIGH_LEVEL_SUPPLY_PATTERN`, `POSSIBLE_DISTRIBUTION_PATTERN`, and `FALSE_BREAKOUT_PATTERN`.
- [ ] Test each pattern with a minimal deterministic label sequence.
- [ ] Run: `python -m pytest tests/test_sequence_patterns.py -v`.

## Task 9: Implement Structure State Classification

**Files:**
- Create: `vpa_structure_recognizer/state_classifier.py`
- Test: `tests/test_state_classifier.py`

- [ ] Convert trend context plus sequence patterns into per-window states.
- [ ] Resolve final state across windows using config weights and confidence rules.
- [ ] Emit `main_features`, `risk_flags`, `bullish_confirm_condition`, and `bearish_invalidate_condition`.
- [ ] Enforce that `POSSIBLE_ACCUMULATION` and `POSSIBLE_DISTRIBUTION` require multi-window sequence evidence plus trend/position context.
- [ ] Run: `python -m pytest tests/test_state_classifier.py -v`.

## Task 10: Implement Top-Down Ranking

**Files:**
- Create: `vpa_structure_recognizer/top_down_ranker.py`
- Test: `tests/test_top_down_ranker.py`

- [ ] Score all-A market state from trend, breadth, amount state, strong-stock ratio, weak-stock ratio, and limit-down risk.
- [ ] Score sectors from sector trend, sector sequence, relative strength versus all-A, and member resonance.
- [ ] Score stocks from self structure, relative strength versus sector, multi-window resonance, and risk penalties.
- [ ] Apply downgrade rules when `market_score < 40` or `sector_score < 40`; mark strong stocks in weak sectors as watch-only.
- [ ] Convert final score to ratings `A` through `E`.
- [ ] Run: `python -m pytest tests/test_top_down_ranker.py -v`.

## Task 11: Implement Backtest Validation Fields

**Files:**
- Create: `vpa_structure_recognizer/backtest_validator.py`
- Test: `tests/test_backtest_validator.py`

- [ ] Compute future returns for `1d`, `3d`, `5d`, `10d`, and `20d`.
- [ ] Compute future max gain/drawdown for `10d` and `20d`.
- [ ] Compute `hit_new_high_20d`, `hit_new_low_20d`, `outperform_sector_10d`, and `outperform_market_10d`.
- [ ] Join validation metrics back to structure states without lookahead in feature or label generation.
- [ ] Run: `python -m pytest tests/test_backtest_validator.py -v`.

## Task 12: Implement Excel Export and Text Reports

**Files:**
- Create: `vpa_structure_recognizer/excel_exporter.py`
- Create: `vpa_structure_recognizer/report_formatter.py`
- Test: `tests/test_excel_exporter.py`
- Test: `tests/test_report_formatter.py`

- [ ] Generate `vpa_structure_report_YYYYMMDD.xlsx`.
- [ ] Include sheets: all-A structure, sector ranking, strong sector candidates, stock summary, three-level resonance stocks, low-level support stocks, healthy uptrend stocks, high-level supply risk stocks, breakdown risk stocks, multi-window label details, and validation data.
- [ ] Format single-stock text reports using the seven report sections from spec section 12.
- [ ] Test workbook sheet names and required columns using a small fixture dataset.
- [ ] Run: `python -m pytest tests/test_excel_exporter.py tests/test_report_formatter.py -v`.

## Task 13: Implement Pipeline CLI

**Files:**
- Create: `vpa_structure_recognizer/pipeline.py`
- Create: `scripts/run_vpa_structure.py`
- Test: `tests/test_pipeline_smoke.py`

- [ ] Implement CLI arguments: `--config`, `--start-date`, `--end-date`, `--as-of-date`, `--source`, `--output-db`, and `--output-dir`.
- [ ] Run pipeline stages in order: load source data, optional local aggregates, features, trend context, bar labels, sequence stats, state classification, top-down ranking, validation, export.
- [ ] Support batch date ranges for 2022, 2023, and 2024.
- [ ] Test a smoke run against fixture DuckDB data and verify all `vpa_*` tables receive rows.
- [ ] Run: `python -m pytest tests/test_pipeline_smoke.py -v`.

## Task 14: External DuckDB Integration Smoke Test

**Files:**
- Create: `tests/test_external_duckdb_contract.py`

- [ ] Add opt-in tests skipped unless `VPA_RUN_EXTERNAL_DUCKDB_TESTS=1`.
- [ ] Verify read-only connection to `/home/nan/alpha-find-v2/output/research_source.duckdb`.
- [ ] Verify required source tables and columns exist.
- [ ] Verify a small date/code slice can normalize into the stock-bar contract.
- [ ] Run manually: `VPA_RUN_EXTERNAL_DUCKDB_TESTS=1 python -m pytest tests/test_external_duckdb_contract.py -v`.

## Task 15: Documentation and Contributor Updates

**Files:**
- Create: `README.md`
- Modify: `AGENTS.md`

- [ ] Document source-data ownership boundary.
- [ ] Document config keys and default paths.
- [ ] Document normal run, smoke test run, external DuckDB contract test, and output locations.
- [ ] Update `AGENTS.md` if final source layout differs from the current contributor guide.
- [ ] Run: `python -m pytest tests -v`.

## Verification Sequence

Run these before claiming implementation completion:

1. `python -m pytest tests -v`
2. `python scripts/run_vpa_structure.py --config config/default.toml --start-date 2024-01-01 --end-date 2024-01-31 --output-db outputs/vpa_smoke.duckdb --output-dir outputs/reports`
3. Inspect generated `outputs/vpa_smoke.duckdb` for all `vpa_*` tables.
4. Inspect generated Excel workbook for required sheet names and non-empty stock/sector/market sheets.
5. Optional external source contract: `VPA_RUN_EXTERNAL_DUCKDB_TESTS=1 python -m pytest tests/test_external_duckdb_contract.py -v`.

## Implementation Order

Recommended order:

1. Tasks 1 to 3 establish config, contracts, and storage.
2. Tasks 4 to 8 build deterministic analytical primitives.
3. Tasks 9 to 11 build interpretation, ranking, and validation.
4. Tasks 12 to 13 build user-facing outputs and orchestration.
5. Tasks 14 to 15 lock the external integration and documentation.

The most parallelizable lanes after Task 3 are: feature/trend/label logic, storage/reporting, and validation/ranking tests. Avoid parallel edits to shared model/config files after the contracts are established.
