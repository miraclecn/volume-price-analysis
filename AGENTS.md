# Repository Guidelines

## Project Structure & Module Organization

The root contains the domain spec, package code, SQL schema, tests, and run scripts.

- `vpa_structure_recognizer/` contains the Python pipeline modules: data adapters, feature engineering, trend context, labels, sequences, state classification, ranking, validation, export, and orchestration.
- `sql/create_vpa_tables.sql` defines project-owned `vpa_*` DuckDB tables.
- `config/default.toml` stores windows, thresholds, scoring weights, source paths, and output paths.
- `scripts/run_vpa_structure.py` is the CLI entrypoint.
- `tests/` contains pytest coverage for each pipeline layer.
- `outputs/` is for generated DuckDB files and reports and must remain untracked.

## Build, Test, and Development Commands

- `python -m pytest tests -v` runs the full test suite.
- `python -m pytest tests/test_features.py -v` runs focused feature tests.
- `python scripts/run_vpa_structure.py --config config/default.toml --start-date 2024-01-01 --end-date 2024-01-31 --source /home/nan/alpha-find-v2/output/research_source.duckdb --output-db outputs/vpa_smoke.duckdb --output-dir outputs/reports` runs a local pipeline batch.
- `VPA_RUN_EXTERNAL_DUCKDB_TESTS=1 python -m pytest tests/test_external_duckdb_contract.py -v` checks the temporary external DuckDB contract.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and clear `snake_case` names for files, functions, variables, and tests. Keep modules aligned with one pipeline responsibility each.

All price movement calculations must use percentage-based features, not raw price differences. Volume strength must be calculated per active window (`window_n`), not from one fixed average.

## Testing Guidelines

Use `pytest`. Name test files `test_*.py` and test functions `test_<behavior>()`. Prioritize regression tests for multi-window labels, percentage features, parent-window trend context, sequence-only stage conclusions, top-down ranking downgrades, and Excel output sheets.

## Commit & Pull Request Guidelines

Use concise, imperative commit subjects that explain why the change exists. Include evidence in the body when useful, for example `Tested: python -m pytest tests -v`.

Pull requests should include a short purpose statement, changed modules, test evidence, and sample output paths or screenshots for report/export changes.

## Security & Configuration Tips

Do not commit credentials, proprietary market data, generated DuckDB files, or bulky reports. Keep thresholds and window definitions in `config/default.toml`; avoid hardcoding strategy parameters inside analysis modules. External source DuckDB files are read-only integration inputs. This project identifies structure for review and validation, not automatic trading signals.
