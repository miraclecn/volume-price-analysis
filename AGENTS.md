# Repository Guidelines

## Project Structure & Module Organization

This repository is currently spec-first. The root contains `A股多层级量价结构识别系统_SPEC.md`, which defines the target behavior for an A-share multi-level volume-price structure recognizer.

When implementation begins, follow the spec's proposed module layout under `modules/research/vpa_structure_recognizer/`:

- `src/` for Python modules such as `feature_engineering.py`, `bar_labeler.py`, `sequence_analyzer.py`, and `excel_exporter.py`.
- `sql/` for table creation, data loading, aggregation, and report export SQL.
- `tests/` for unit and regression tests.
- `outputs/reports/` and `outputs/validation/` for generated Excel reports and validation artifacts. Do not commit large generated outputs unless explicitly required.

## Build, Test, and Development Commands

No build system or runnable package is committed yet. Once code is scaffolded, prefer standard Python commands:

- `python -m pytest tests` runs the test suite.
- `python -m pytest tests/test_features.py` runs a focused test file.
- `python modules/research/vpa_structure_recognizer/run_vpa_structure.py --config modules/research/vpa_structure_recognizer/config.yaml` runs the recognizer, once the entrypoint exists.

Add project-specific commands to `README.md` as soon as they are introduced.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and clear snake_case names for files, functions, variables, and test cases. Keep modules aligned with the spec's pipeline: data loading, feature engineering, trend context, bar labeling, sequence analysis, state classification, ranking, validation, and export.

All price movement calculations must use percentage-based features, not raw price differences. Volume strength must be calculated per active window (`window_n`), not from one fixed average.

## Testing Guidelines

Use `pytest`. Name test files `test_*.py` and test functions `test_<behavior>()`. Prioritize regression tests for the spec's invariants: multi-window labels per trading day, percentage-based price features, parent-window trend context, top-down market-to-sector-to-stock ranking, and Excel output fields.

## Commit & Pull Request Guidelines

This directory is not currently a Git repository, so no local commit history is available. Use concise, imperative commit subjects that explain why the change exists. Include evidence in the body when useful, for example `Tested: python -m pytest tests`.

Pull requests should include a short purpose statement, changed modules, test evidence, and sample output paths or screenshots for report/export changes.

## Security & Configuration Tips

Do not commit credentials, proprietary market data, or bulky generated reports. Keep thresholds and window definitions in `config.yaml`; avoid hardcoding strategy parameters inside analysis modules. This project identifies structure for review and validation, not automatic trading signals.
