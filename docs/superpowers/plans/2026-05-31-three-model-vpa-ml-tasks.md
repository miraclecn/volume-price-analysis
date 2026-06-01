# Three-Model VPA-ML Incremental Task Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade this repository's `ml_stock_selector` path to the three-model design from `三模型量价学习规范说明.md`, excluding all `alpha-data` repository changes.

**Architecture:** Keep `vpa_structure_recognizer` and existing `vpa_*` table semantics unchanged. Add v2 behavior behind config flags in `ml_stock_selector`: no industry metadata in training features, v2 active-return labels, Absolute Ranker + Active Ranker + true Risk Model, v2 scoring, and legacy-compatible daily signal / portfolio / backtest paths.

**Tech Stack:** Python, pandas, DuckDB, LightGBM or existing fallback model wrappers, scikit-learn metrics, pytest.

---

## Scope Boundary

Included:

- Files under `ml_stock_selector/`
- ML scripts under `scripts/`
- ML schema under `sql/create_ml_tables.sql`
- ML config under `config/ml_default.toml`
- ML docs under `README.md` and `docs/ml_stock_selector_operating_notes.md`
- Tests under `tests/test_ml_*`, `tests/test_alpha_ranker.py`, `tests/test_risk_model.py`, `tests/test_daily_signal.py`, `tests/test_portfolio_constructor.py`

Excluded:

- Any implementation in the `alpha-data` repository
- `docs/vpa_ml_consumer_contract.md` in `alpha-data`
- `alpha_data_local/*`
- Creation of upstream `market_benchmark_daily` or `industry_benchmark_daily` inside `alpha-data`

Important local assumption:

- Until upstream benchmark views exist, this project should compute market and industry benchmark returns locally from `stock_bar_normalized_daily`, while keeping loaders compatible with optional upstream benchmark tables later.

## Delivery Order

| Phase | Tasks | Exit Criteria |
|---|---:|---|
| P0 compatibility foundation | 1-6 | v2 flags exist; schema is additive; no-industry feature path and active labels are tested while legacy path still works |
| P1 three-model training and prediction | 7-12 | absolute, active, and risk artifacts can be trained, registered, loaded, and used for batch prediction |
| P2 scoring, portfolio, serving | 13-16 | v2 scores, candidate/core pools, cash gating, and daily signal work with old path preserved |
| P3 docs and verification | 17-18 | docs and smoke tests describe and prove the migration path |

---

### Task 1: Add v2 Feature Flags And Config Plumbing

**Files:**

- Modify: `config/ml_default.toml`
- Modify: `ml_stock_selector/config.py`
- Modify: `ml_stock_selector/constants.py`
- Test: `tests/test_config.py`
- Test: `tests/test_ml_config.py`

**Steps:**

- [ ] Add an `[ml_v2]` or equivalent config section with all v2 flags defaulting to `false`: `exclude_industry_metadata_from_features_json`, `feature_matrix_v2_deny_industry`, `labels_v2_enabled`, `active_ranker_enabled`, `risk_model_v2_enabled`, `trade_score_v2_enabled`, `daily_signal_v2_enabled`.
- [ ] Add optional score/pool thresholds in config with conservative defaults from the spec: absolute candidate `0.70`, active candidate `0.70`, risk candidate max `0.60`, absolute core `0.80`, active core `0.75`, risk core max `0.35`, min v2 trade score `0.80`.
- [ ] Update `MLConfig` loading so missing v2 sections fall back to legacy defaults instead of failing older configs.
- [ ] Add constants for `MODEL_TYPE_ACTIVE_RANKER`, `SCORE_VERSION_LEGACY`, `SCORE_VERSION_THREE_MODEL`, and `FEATURE_SCHEMA_V2_NO_INDUSTRY`.
- [ ] Test that existing `config/ml_default.toml` loads and all v2 flags are false unless explicitly enabled.

**Acceptance Criteria:**

- `load_ml_config("config/ml_default.toml")` succeeds.
- Legacy tests still see the old behavior by default.
- A test config with v2 flags enabled exposes typed values without changing unrelated settings.

---

### Task 2: Extend ML Tables Additively

**Files:**

- Modify: `sql/create_ml_tables.sql`
- Modify if present: `ml_stock_selector/contracts/ml_schema.py`
- Test: `tests/test_ml_storage_schema.py`
- Test: `tests/test_ml_contracts.py`

**Steps:**

- [ ] Add nullable v2 columns to `ml_labels_daily`: `absolute_ret`, `absolute_rank_pct`, `absolute_label`, `market_ret`, `industry_ret`, `market_excess_ret`, `industry_excess_ret`, `active_score`, `active_rank_pct`, `active_label`, `benchmark_missing_market`, `benchmark_missing_industry`, `benchmark_peer_count`.
- [ ] Add nullable v2 columns to `ml_predictions_daily`: `absolute_score`, `absolute_rank_pct`, `absolute_zscore`, `active_score`, `active_rank_pct`, `active_zscore`, `risk_prob`, `risk_zscore`, `core_score`, `trade_score_v2`, `score_version`.
- [ ] Keep existing columns and primary keys unchanged.
- [ ] Add local optional tables if needed for caching computed benchmark returns: `ml_market_benchmark_daily` and `ml_industry_benchmark_daily`.
- [ ] Update schema tests to assert new columns exist and old inserts still work with only legacy columns.

**Acceptance Criteria:**

- Existing upserts into `ml_labels_daily` and `ml_predictions_daily` still succeed.
- New v2 columns are nullable.
- Primary keys remain backward-compatible.

---

### Task 3: Keep Industry Metadata Out Of `features_json`

**Files:**

- Modify: `ml_stock_selector/feature_mart.py`
- Test: `tests/test_ml_feature_mart.py`
- Test: `tests/test_ml_unknown_industry.py`

**Steps:**

- [ ] Keep `industry_code` and `industry_name` as table-level metadata columns in `ml_feature_mart_daily`.
- [ ] When `exclude_industry_metadata_from_features_json=true`, build `features_json` only from OHLCV and VPA feature families.
- [ ] Exclude `industry_code`, `industry_name`, and `industry_unknown` from v2 JSON.
- [ ] Preserve legacy JSON behavior when the flag is false.
- [ ] Add a regression test that v2 `features_json` has no industry keys while metadata columns remain populated.

**Acceptance Criteria:**

- `features_json` does not include `industry_code`, `industry_name`, or `industry_unknown` in v2.
- UNKNOWN industry rows remain present in the feature mart and can still be constrained at portfolio time.

---

### Task 4: Add `feature_matrix` Industry Denylist

**Files:**

- Modify: `ml_stock_selector/feature_matrix.py`
- Test: `tests/test_feature_matrix.py`
- Test: `tests/test_ml_unknown_industry.py`

**Steps:**

- [ ] Add a denylist that excludes `industry_code`, `industry_name`, `industry_unknown`, and future `industry_*` fields before schema fitting.
- [ ] Apply the denylist in both fit and transform paths when the v2 flag or v2 schema version is active.
- [ ] Persist `schema_version="v2_no_industry"` for v2 schemas.
- [ ] Ensure non-industry categorical VPA fields still receive `__MISSING__` and `__UNKNOWN__` one-hot handling.
- [ ] Add a test using a legacy mart JSON that contains industry fields; v2 matrix output must contain no `industry_*` or `industry_code=...` columns.

**Acceptance Criteria:**

- v2 schema output columns contain zero industry-derived columns even with old JSON inputs.
- Legacy v1 schema behavior remains available.

---

### Task 5: Implement Local Benchmark Return Builder

**Files:**

- Modify: `ml_stock_selector/label_builder.py`
- Modify: `ml_stock_selector/data_access.py`
- Test: `tests/test_ml_label_builder.py`

**Steps:**

- [ ] Add local market benchmark return calculation for each `trade_date`, `horizon_d`, and `label_base`, using the same future window and base price convention as stock labels.
- [ ] Add industry benchmark return calculation by `industry_code`, excluding `UNKNOWN` as a real industry benchmark.
- [ ] Include `benchmark_peer_count`; when peer count is below the configured minimum or industry is UNKNOWN, set `industry_ret=NULL` and `benchmark_missing_industry=true`.
- [ ] Keep optional loader hooks for future upstream benchmark tables, but do not require those tables for local tests.
- [ ] Add tests for market up 5 percent versus stock up 5 percent giving `market_excess_ret ~= 0`, and market down 3 percent versus stock up 1 percent giving `market_excess_ret ~= 0.04`.

**Acceptance Criteria:**

- Benchmark labels can be generated from only `stock_bar_normalized_daily`-contract data.
- UNKNOWN industry rows still receive market-relative active labels.
- No `alpha-data` benchmark table is required.

---

### Task 6: Add v2 Active And Absolute Label Columns

**Files:**

- Modify: `ml_stock_selector/label_builder.py`
- Modify: `scripts/build_ml_labels.py`
- Test: `tests/test_ml_label_builder.py`
- Test: `tests/test_ml_storage_schema.py`

**Steps:**

- [ ] Add `absolute_ret` as the explicit alias for current `future_ret`.
- [ ] Add `absolute_rank_pct` and `absolute_label`, compatible with current `future_rank_pct` and `rank_label`.
- [ ] Add `market_excess_ret`, `industry_excess_ret`, `active_score`, `active_rank_pct`, and `active_label`.
- [ ] Calculate `active_score = market_excess_ret` when industry benchmark is missing; otherwise use `0.5 * market_excess_ret + 0.5 * industry_excess_ret`.
- [ ] Keep legacy columns populated for all rows.
- [ ] Gate v2 column generation through `labels_v2_enabled` in scripts while allowing direct unit tests to call the v2 builder.

**Acceptance Criteria:**

- Old label consumers still find `future_ret`, `future_score`, `future_rank_pct`, `rank_label`, and `risk_label`.
- Active Ranker training can use non-null `active_label` for rows with valid market benchmark data.

---

### Task 7: Make Training Sample Selection Label-Aware

**Files:**

- Modify: `ml_stock_selector/sample_builder.py`
- Test: `tests/test_sample_builder.py`
- Test: `tests/test_ml_datasets.py`

**Steps:**

- [ ] Replace hardcoded `dropna(subset=["rank_label", "future_score"])` with label-aware validation.
- [ ] Allow callers to request `rank_label`, `absolute_label`, `active_label`, or `risk_label`.
- [ ] For ranker labels, require the selected label and its corresponding target return or score column.
- [ ] For risk labels, require `risk_label` only plus features.
- [ ] Add tests proving active and risk samples are not silently dropped by legacy rank-label assumptions.

**Acceptance Criteria:**

- `build_training_samples(..., label_name="active_label")` returns rows with valid active labels.
- `build_training_samples(..., label_name="risk_label")` works without `future_score`.

---

### Task 8: Upgrade Absolute Ranker Outputs And Metrics

**Files:**

- Modify: `ml_stock_selector/models/alpha_ranker.py`
- Modify if needed: `ml_stock_selector/models/artifacts.py`
- Test: `tests/test_alpha_ranker.py`

**Steps:**

- [ ] Treat existing `alpha_ranker.py` as the Absolute Ranker implementation.
- [ ] Train on `absolute_label` when present, falling back to `rank_label` in legacy mode.
- [ ] Configure LightGBM with `eval_at=[10, 15]` when available.
- [ ] Add RankIC-style training metric using cross-sectional rank correlation by `trade_date`.
- [ ] Keep model loading compatible with existing artifacts.

**Acceptance Criteria:**

- Absolute Ranker can train with either legacy `rank_label` or v2 `absolute_label`.
- Metrics include NDCG-compatible metadata and RankIC-style evidence when possible.

---

### Task 9: Add Active Ranker

**Files:**

- Create: `ml_stock_selector/models/active_ranker.py`
- Modify: `ml_stock_selector/models/__init__.py`
- Modify: `ml_stock_selector/constants.py`
- Test: `tests/test_active_ranker.py`
- Test: `tests/test_alpha_ranker.py`

**Steps:**

- [ ] Reuse the shared ranker training path from Absolute Ranker where practical.
- [ ] Train with `active_label`.
- [ ] Use cross-sectional grouping by `trade_date`.
- [ ] Persist a distinct model type, e.g. `active_ranker`.
- [ ] Expose a loader/predictor that returns raw active scores for later percentile and z-score normalization.

**Acceptance Criteria:**

- Active Ranker artifact registers independently from Absolute Ranker.
- Active Ranker predictions can be converted to `active_rank_pct` by trade date.

---

### Task 10: Rewrite Risk Model As A True Classifier

**Files:**

- Modify: `ml_stock_selector/models/risk_model.py`
- Modify if needed: `ml_stock_selector/models/artifacts.py`
- Test: `tests/test_risk_model.py`

**Steps:**

- [ ] Stop wrapping `train_alpha_ranker()` as the risk model.
- [ ] Train a binary classifier on `risk_label`.
- [ ] Prefer LightGBM classifier when available; keep a deterministic fallback for test environments.
- [ ] Add `predict_proba` style serving so downstream gets `risk_prob`.
- [ ] Calculate ROC AUC when both classes are present; otherwise record an explicit insufficient-class metric.

**Acceptance Criteria:**

- Risk predictions include probabilities in `[0, 1]`.
- `risk_rank_pct` ranks higher-risk stocks as more dangerous.
- Tests prove risk model no longer reuses ranker model type or ranker objective.

---

### Task 11: Train And Register Three Model Roles

**Files:**

- Modify: `scripts/train_ml_models.py`
- Modify: `ml_stock_selector/registry.py`
- Test: `tests/test_ml_registry.py`
- Test: `tests/test_alpha_ranker.py`
- Test: `tests/test_risk_model.py`
- Test: `tests/test_active_ranker.py`

**Steps:**

- [ ] Keep legacy single-ranker training when v2 flags are off.
- [ ] When v2 flags are on, train Absolute Ranker, Active Ranker, and Risk Model according to enabled flags.
- [ ] Register each model with distinct `model_type`, `label_name`, `label_base`, `horizon_d`, schema URI, artifact URI, params, and metrics.
- [ ] Activate models independently by model role.
- [ ] Print or log all trained model IDs for downstream scripts.

**Acceptance Criteria:**

- A v2 training run can produce three active artifacts for the same `feature_set_id`, `label_base`, and `horizon_d`.
- Legacy `train_ml_models.py` behavior remains available.

---

### Task 12: Generate Combined Three-Model Predictions

**Files:**

- Modify: `ml_stock_selector/prediction.py`
- Modify: `ml_stock_selector/serving/artifact_loader.py`
- Modify: `scripts/run_ml_batch_predict.py`
- Test: `tests/test_ml_prediction.py`

**Steps:**

- [ ] Add loaders for active ranker and risk model artifacts.
- [ ] Add prediction helpers that calculate `absolute_score`, `absolute_rank_pct`, `absolute_zscore`, `active_score`, `active_rank_pct`, `active_zscore`, `risk_prob`, `risk_rank_pct`, and `risk_zscore`.
- [ ] Preserve old `alpha_score` / `alpha_rank_pct` output in legacy mode.
- [ ] Use a legacy-compatible combined row identity for v2 predictions without changing table primary keys.
- [ ] Upsert v2 prediction rows into the expanded `ml_predictions_daily`.

**Acceptance Criteria:**

- Batch prediction writes all three model outputs for each date/code.
- Legacy batch prediction still writes old fields.

---

### Task 13: Implement v2 Trade Scoring

**Files:**

- Modify: `ml_stock_selector/scoring.py`
- Test: `tests/test_ml_scoring.py`

**Steps:**

- [ ] Keep current `score_candidates()` formula as v1 behavior.
- [ ] Add a v2 scorer with `core_score = 0.50 * absolute_rank_pct + 0.35 * active_rank_pct - 0.15 * risk_rank_pct`.
- [ ] Add `trade_score_v2 = core_score + 0.10 * liquidity_score_pct + overlay_context - penalty_score`.
- [ ] Default `overlay_context` to zero or an explicitly configured VPA-only overlay; do not use industry category fields.
- [ ] Populate `score_version` with `v1_legacy` or `v2_three_model`.

**Acceptance Criteria:**

- v1 and v2 scores can be computed side by side.
- Tests cover higher active score improving v2 score and higher risk rank reducing it.

---

### Task 14: Add Candidate/Core Pool Construction And Cash Gating

**Files:**

- Modify: `ml_stock_selector/portfolio/constraints.py`
- Modify: `ml_stock_selector/portfolio/constructor.py`
- Test: `tests/test_portfolio_constructor.py`
- Test: `tests/test_ml_unknown_industry.py`

**Steps:**

- [ ] Keep existing hard filters: ST, pause, `can_buy_next_open`, ADV threshold, and `min_trade_score`.
- [ ] Add candidate pool logic: `(absolute_rank_pct >= 0.70 or active_rank_pct >= 0.70) and risk_rank_pct <= 0.60`.
- [ ] Add core pool logic: `absolute_rank_pct >= 0.80`, `active_rank_pct >= 0.75`, `risk_rank_pct <= 0.35`, `trade_score_v2 >= 0.80`.
- [ ] Allow empty target output when candidate count is too low, core pool is empty, or core median score is below threshold.
- [ ] Preserve `max_unknown_industry_names=1` default and industry diversification constraints.

**Acceptance Criteria:**

- Low-quality v2 prediction days can intentionally produce an empty portfolio.
- UNKNOWN industry candidates can be scored but are capped in final holdings.

---

### Task 15: Upgrade Daily Signal To Load Three Artifacts

**Files:**

- Modify: `ml_stock_selector/serving/daily_signal.py`
- Modify: `scripts/run_ml_daily_signal.py`
- Test: `tests/test_daily_signal.py`

**Steps:**

- [ ] Keep current single-ranker daily signal when `daily_signal_v2_enabled=false`.
- [ ] When v2 is enabled, load active Absolute Ranker, Active Ranker, and Risk Model artifacts.
- [ ] Generate combined v2 prediction rows, score candidates with v2 scoring, and build candidate/core portfolio targets.
- [ ] Upsert predictions and targets with v2 fields populated.
- [ ] Annotate exclusion reasons for unknown industry limit, insufficient candidate/core pool, and score threshold where practical.

**Acceptance Criteria:**

- Daily signal output includes absolute, active, risk, core, and final v2 trade scores.
- Existing daily signal tests for the legacy path still pass.

---

### Task 16: Wire v2 Scoring Into Backtest And Reports

**Files:**

- Modify: `scripts/run_ml_backtest.py`
- Modify: `ml_stock_selector/backtest/engine.py`
- Modify: `ml_stock_selector/backtest/metrics.py`
- Modify: `ml_stock_selector/backtest/reports.py`
- Test: `tests/test_backtest_engine.py`
- Test: `tests/test_backtest_metrics.py`

**Steps:**

- [ ] Let backtest choose `trade_score` or `trade_score_v2` by config score version.
- [ ] Track `candidate_pool_size`, `core_pool_size`, and `cash_days_ratio`.
- [ ] Keep T+1 execution assumptions unchanged.
- [ ] Add report fields for `score_version`, model IDs, and v2 risk/active score summaries.
- [ ] Keep legacy report output working.

**Acceptance Criteria:**

- Backtest can run on v2 predictions without schema errors.
- Cash days are treated as valid strategy outcomes, not failures.

---

### Task 17: Update Operating Documentation

**Files:**

- Modify: `README.md`
- Modify: `docs/ml_stock_selector_operating_notes.md`

**Steps:**

- [ ] Document the three model roles and their labels.
- [ ] State explicitly that industry metadata is allowed as table metadata but not as model input features.
- [ ] Document v2 flags and their default-off migration behavior.
- [ ] Add the updated command chain: feature mart, labels v2, three-model training, batch prediction, backtest, daily signal.
- [ ] Document that alpha-data benchmark views are optional and this repo has local benchmark fallback logic.

**Acceptance Criteria:**

- A reader can run either the legacy path or v2 path from docs.
- Docs do not assign implementation work to `alpha-data`.

---

### Task 18: Add End-To-End v2 Verification

**Files:**

- Modify: `tests/test_ml_pipeline_smoke.py`
- Modify: `tests/test_daily_signal.py`
- Modify: `tests/test_ml_storage_schema.py`

**Steps:**

- [ ] Build a fixture with normalized bars, VPA tables, known industry metadata, and at least one UNKNOWN industry row.
- [ ] Generate v2 feature mart with no industry JSON features.
- [ ] Generate v2 labels with active labels and benchmark missing flags.
- [ ] Train Absolute Ranker, Active Ranker, and Risk Model using the fixture.
- [ ] Run v2 prediction, scoring, and portfolio construction.
- [ ] Assert the pipeline can produce either selected targets or an intentional empty portfolio with `allow_cash=true`.

**Acceptance Criteria:**

- Focused smoke test passes without requiring external `alpha-data` benchmark views.
- Verification command:

```bash
python -m pytest \
  tests/test_ml_feature_mart.py \
  tests/test_feature_matrix.py \
  tests/test_ml_label_builder.py \
  tests/test_alpha_ranker.py \
  tests/test_active_ranker.py \
  tests/test_risk_model.py \
  tests/test_ml_scoring.py \
  tests/test_portfolio_constructor.py \
  tests/test_daily_signal.py \
  tests/test_ml_pipeline_smoke.py \
  -v
```

---

## Explicit Non-Tasks From The Source Spec

These items are described in `三模型量价学习规范说明.md`, but are outside this repository's task scope:

- Updating `alpha-data/docs/vpa_ml_consumer_contract.md`
- Adding benchmark views inside `alpha-data`
- Changing `alpha_data_local/research_source_contract.py`
- Changing `alpha_data_local/market_data_bootstrap.py`
- Changing `alpha_data_local/market_data_quality.py`
- Changing `alpha_data_local/cli.py`
- Updating `alpha-data` tests for market-data quality, industry contract, or normalized bars

This project may only consume those upstream capabilities when available, or compute local fallbacks from the normalized-bar contract.
