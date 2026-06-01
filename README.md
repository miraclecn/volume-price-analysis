# Volume Price Analysis

This repository implements the A-share multi-level volume-price structure recognizer described in `A股多层级量价结构识别系统_SPEC.md`.

## ML Stock Selector Subsystem

`ml_stock_selector` is a downstream ML subsystem. It consumes alpha-data
`stock_bar_normalized_daily` plus this repository's `vpa_*` tables, and writes
only `ml_*` tables and model artifacts under `outputs/ml/`.

The intended command sequence is:

```bash
python scripts/run_alpha_data_contract_check.py --db /home/nan/alpha-data-local/output/research_source.duckdb
python scripts/run_ml_schema_check.py --vpa-db outputs/vpa.duckdb
python scripts/run_ml_feature_mart.py --config config/ml_default.toml
python scripts/build_ml_labels.py --config config/ml_default.toml
python scripts/train_ml_models.py --config config/ml_default.toml
python scripts/run_ml_batch_predict.py --config config/ml_default.toml
python scripts/run_ml_backtest.py --config config/ml_default.toml
python scripts/run_ml_daily_signal.py --config config/ml_default.toml --as-of-date YYYY-MM-DD
```

Backtests and daily signals use T-day information for decisions and execute no
earlier than T+1. Feature sets are ordered from `baseline_a_ohlcv` through
`vpa_e_structure_state`; high-level VPA states are isolated to the final
ablation set.

VPA-ML consumes alpha-data's UNKNOWN industry standard without repair. Training
keeps UNKNOWN samples, portfolio construction defaults to at most one UNKNOWN
holding through `max_unknown_industry_names = 1`, UNKNOWN is not treated as
untradable by itself, and backtest reports track UNKNOWN exposure separately.

The optional v2 ML path uses three model roles: Absolute Ranker
(`absolute_label`), Active Ranker (`active_label` from market/industry excess
returns), and Risk Model (`risk_label` probability). All v2 behavior is gated by
`[ml_v2]` flags in `config/ml_default.toml`, and the default config currently
opens those flags. In v2, industry code/name are retained as
metadata for reports and portfolio constraints, but are excluded from
`features_json` and denied again in `feature_matrix` so they cannot become model
features. Market and industry benchmark labels are computed locally from
`stock_bar_normalized_daily` unless upstream benchmark views are available later.

## Scope

This project does not prepare raw market data. Upstream projects own downloads, qfq adjustment, PIT reference construction, ST/suspension/limit repair, and permanent source marts. This project reads those prepared DuckDB files through read-only adapters, then writes project-owned `vpa_*` derived tables, validation metrics, and reports.

VPA consumes alpha-data's normalized stock-bar contract from
`stock_bar_normalized_daily`. Industry normalization belongs to alpha-data:
when alpha-data cannot classify a stock it emits `industry_code = "UNKNOWN"`
and `industry_name = "UNKNOWN"`. VPA keeps those stocks in stock-scope
analysis, may build an observational `UNKNOWN` sector aggregate, and excludes
`UNKNOWN` from real industry strength context by assigning affected stocks a
neutral sector score with an `industry_unknown` risk flag. VPA does not read or
repair alpha-data's raw industry classification tables.

Default source path is configured in `config/default.toml`:

- `/home/nan/alpha-data-local/output/research_source.duckdb`

## Commands

Run all tests:

```bash
python -m pytest tests -v
```

Run the pipeline against a source DuckDB:

```bash
python scripts/run_vpa_structure.py \
  --config config/default.toml \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --source /home/nan/alpha-data-local/output/research_source.duckdb \
  --output-db outputs/vpa_smoke.duckdb \
  --output-dir outputs/reports
```

Run the opt-in external source contract test:

```bash
VPA_RUN_EXTERNAL_DUCKDB_TESTS=1 python -m pytest tests/test_external_duckdb_contract.py -v
```

## Outputs

The pipeline writes DuckDB tables:

- `vpa_features`
- `vpa_trend_context`
- `vpa_bar_context_labels`
- `vpa_sequence_stats`
- `vpa_structure_state`

It also writes `vpa_structure_report_YYYYMMDD.xlsx` under the configured report directory.

## Development Notes

Keep all generated data under `outputs/`. Do not write derived data back into upstream DuckDB files. Add tests before changing analytical rules, especially for percentage feature calculations, parent-window trend context, sequence patterns, and top-down rating downgrades.
