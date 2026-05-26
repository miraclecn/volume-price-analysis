# Volume Price Analysis

This repository implements the A-share multi-level volume-price structure recognizer described in `A股多层级量价结构识别系统_SPEC.md`.

## Scope

This project does not prepare raw market data. Upstream projects own downloads, qfq adjustment, PIT reference construction, ST/suspension/limit repair, and permanent source marts. This project reads those prepared DuckDB files through read-only adapters, then writes project-owned `vpa_*` derived tables, validation metrics, and reports.

Default temporary source paths are configured in `config/default.toml`:

- `/home/nan/alpha-find-v2/output/research_source.duckdb`
- `/home/nan/alpha-find/output/stock_data_audited.duckdb`

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
  --source /home/nan/alpha-find-v2/output/research_source.duckdb \
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
